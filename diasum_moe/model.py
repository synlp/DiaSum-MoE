import copy
import inspect
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .config import ModelSettings
from .data import DialogueBatch


@dataclass
class RoutingOutput:
    scores: list[torch.Tensor]
    expert_indices: list[torch.Tensor]


@dataclass
class SelectedExpertOutput:
    batch_index: int
    utterance_index: int
    rank: int
    expert_index: int
    hidden_states: torch.Tensor


@dataclass
class DiaSumMoEOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    routing: RoutingOutput
    selected_experts: list[list[SelectedExpertOutput]]
    condition_lengths: list[int]


def _layer_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("Decoder layer returned unsupported output")


def validated_layers(decoder: nn.Module, expected: int, label: str) -> list[nn.Module]:
    layers = list(decoder.layers)
    if len(layers) != expected:
        raise ValueError(f"{label} has {len(layers)} layers; expected {expected}")
    return layers


class RoutingBackbone(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        routing_layers: Iterable[nn.Module],
        expert_template: Iterable[nn.Module],
        final_norm: nn.Module,
        hidden_size: int,
        num_experts: int,
        rotary_embedding: nn.Module | None = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.routing_layers = nn.ModuleList(routing_layers)
        template = nn.ModuleList(expert_template)
        self.experts = nn.ModuleList([copy.deepcopy(template) for _ in range(num_experts)])
        self.expert_norms = nn.ModuleList([copy.deepcopy(final_norm) for _ in range(num_experts)])
        self.hidden_size = hidden_size
        self.rotary_embedding = rotary_embedding

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def _causal_mask(self, attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

            return _prepare_4d_causal_attention_mask(
                attention_mask,
                (hidden_states.shape[0], hidden_states.shape[1]),
                hidden_states,
                0,
            )
        except ImportError:
            length = hidden_states.shape[1]
            causal = torch.full((length, length), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
            return torch.triu(causal, diagonal=1).view(1, 1, length, length)

    def run_stack(
        self,
        layers: Iterable[nn.Module],
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        position_ids = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(sequence_length, device=hidden_states.device)
        causal_mask = self._causal_mask(attention_mask, hidden_states)
        position_embeddings = None
        if self.rotary_embedding is not None:
            position_embeddings = self.rotary_embedding(hidden_states, position_ids)
        for layer in layers:
            parameters = inspect.signature(layer.forward).parameters
            kwargs: dict[str, Any] = {}
            if "attention_mask" in parameters:
                kwargs["attention_mask"] = causal_mask
            if "position_ids" in parameters:
                kwargs["position_ids"] = position_ids
            if "position_embeddings" in parameters and position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            if "cache_position" in parameters:
                kwargs["cache_position"] = cache_position
            if "use_cache" in parameters:
                kwargs["use_cache"] = False
            if "hidden_states" in parameters:
                output = layer(hidden_states=hidden_states, **kwargs)
            else:
                output = layer(hidden_states, **kwargs)
            hidden_states = _layer_hidden(output)
        return hidden_states

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dialogue_embeddings = self.embed(input_ids)
        dialogue_hidden = self.run_stack(self.routing_layers, dialogue_embeddings, attention_mask)
        return dialogue_embeddings, dialogue_hidden

    def process(self, expert_index: int, utterance_hidden: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones(utterance_hidden.shape[:2], device=utterance_hidden.device, dtype=torch.long)
        output = self.run_stack(self.experts[expert_index], utterance_hidden, attention_mask)
        return self.expert_norms[expert_index](output)


class RoleOrientedRouting(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k exceeds num_experts")
        self.projection = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.memory = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.num_experts = num_experts
        self.top_k = top_k
        nn.init.xavier_uniform_(self.projection)
        nn.init.normal_(self.memory, mean=0.0, std=hidden_size**-0.5)

    def forward(self, dialogue_hidden: torch.Tensor, routing_positions: list[list[int]]) -> RoutingOutput:
        score_rows: list[torch.Tensor] = []
        expert_rows: list[torch.Tensor] = []
        for batch_index, positions in enumerate(routing_positions):
            if not positions:
                raise ValueError("Every dialogue requires at least one role position")
            role_hidden = dialogue_hidden[batch_index, torch.tensor(positions, device=dialogue_hidden.device)]
            scores = (role_hidden @ self.projection) @ self.memory.transpose(0, 1)
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            score_rows.append(scores)
            expert_rows.append(order[:, : self.top_k])
        return RoutingOutput(score_rows, expert_rows)


class FusionGenerator(nn.Module):
    def __init__(self, language_model: nn.Module, hidden_size: int, soft_prompt_length: int):
        super().__init__()
        self.language_model = language_model
        self.hidden_size = hidden_size
        self.soft_prompt = nn.Parameter(torch.empty(soft_prompt_length, hidden_size))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=hidden_size**-0.5)

    def token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(input_ids)

    def build_condition(
        self,
        dialogue_embeddings: torch.Tensor,
        selected_experts: list[SelectedExpertOutput],
    ) -> torch.Tensor:
        ordered = sorted(selected_experts, key=lambda item: (item.utterance_index, item.rank))
        pieces = [self.soft_prompt, dialogue_embeddings]
        pieces.extend(item.hidden_states.squeeze(0) for item in ordered)
        return torch.cat(pieces, dim=0)

    def forward(
        self,
        conditions: list[torch.Tensor],
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[int]]:
        target_embeddings = self.token_embeddings(target_ids)
        condition_lengths = [condition.shape[0] for condition in conditions]
        total_lengths = [condition_lengths[index] + int(target_attention_mask[index].sum()) for index in range(len(conditions))]
        max_length = max(total_lengths)
        batch_size = len(conditions)
        device = target_embeddings.device
        dtype = target_embeddings.dtype
        inputs_embeds = torch.zeros((batch_size, max_length, self.hidden_size), device=device, dtype=dtype)
        attention_mask = torch.zeros((batch_size, max_length), device=device, dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, device=device, dtype=torch.long)
        for index, condition in enumerate(conditions):
            condition_length = condition_lengths[index]
            target_length = int(target_attention_mask[index].sum())
            inputs_embeds[index, :condition_length] = condition
            inputs_embeds[index, condition_length : condition_length + target_length] = target_embeddings[index, :target_length]
            attention_mask[index, : condition_length + target_length] = 1
            labels[index, condition_length : condition_length + target_length] = target_ids[index, :target_length]
        output = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
        return output.loss, output.logits, condition_lengths

    def generate(
        self,
        condition: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        inputs_embeds = condition.unsqueeze(0)
        attention_mask = torch.ones((1, condition.shape[0]), device=condition.device, dtype=torch.long)
        generated = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        if generated.shape[1] > max_new_tokens:
            generated = generated[:, -max_new_tokens:]
        return generated


class DiaSumMoE(nn.Module):
    def __init__(
        self,
        routing_backbone: RoutingBackbone,
        fusion_language_model: nn.Module,
        num_experts: int,
        top_k: int,
        soft_prompt_length: int,
    ):
        super().__init__()
        if len(routing_backbone.experts) != num_experts:
            raise ValueError("Expert bank size mismatch")
        self.routing_backbone = routing_backbone
        self.router = RoleOrientedRouting(routing_backbone.hidden_size, num_experts, top_k)
        self.fusion = FusionGenerator(fusion_language_model, routing_backbone.hidden_size, soft_prompt_length)

    @classmethod
    def from_pretrained(
        cls,
        settings: ModelSettings,
        tokenizer_size: int,
        dtype: torch.dtype,
    ) -> "DiaSumMoE":
        from transformers import AutoModelForCausalLM

        routing_model = AutoModelForCausalLM.from_pretrained(
            settings.backbone,
            torch_dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        routing_model.resize_token_embeddings(tokenizer_size)
        decoder = routing_model.get_decoder()
        layers = validated_layers(decoder, settings.total_layers, "Routing backbone")
        split = settings.total_layers - settings.expert_layers
        embedding = decoder.embed_tokens
        final_norm = decoder.norm
        rotary_embedding = getattr(decoder, "rotary_emb", None)
        routing_layers = layers[:split]
        expert_template = layers[split:]
        hidden_size = int(routing_model.config.hidden_size)
        decoder.layers = nn.ModuleList()
        decoder.embed_tokens = nn.Identity()
        decoder.norm = nn.Identity()
        del routing_model
        routing_backbone = RoutingBackbone(
            embedding,
            routing_layers,
            expert_template,
            final_norm,
            hidden_size,
            settings.num_experts,
            rotary_embedding,
        )
        fusion_model = AutoModelForCausalLM.from_pretrained(
            settings.fusion_backbone,
            torch_dtype=dtype,
            trust_remote_code=settings.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        fusion_model.resize_token_embeddings(tokenizer_size)
        fusion_decoder = fusion_model.get_decoder()
        validated_layers(fusion_decoder, settings.total_layers, "Fusion backbone")
        if int(fusion_model.config.hidden_size) != hidden_size:
            raise ValueError("Routing and fusion hidden sizes differ")
        model = cls(routing_backbone, fusion_model, settings.num_experts, settings.top_k, settings.soft_prompt_length)
        return model.to(dtype=dtype)

    def route_and_process(
        self,
        dialogue_hidden: torch.Tensor,
        spans: list[list[tuple[int, int]]],
        routing: RoutingOutput,
    ) -> list[list[SelectedExpertOutput]]:
        selected_by_batch: list[list[SelectedExpertOutput]] = []
        for batch_index, utterance_spans in enumerate(spans):
            selected: list[SelectedExpertOutput] = []
            for utterance_index, (start, end) in enumerate(utterance_spans):
                utterance_hidden = dialogue_hidden[batch_index : batch_index + 1, start:end]
                utterance_hidden = torch.cat((utterance_hidden[:, -1:], utterance_hidden[:, :-1]), dim=1)
                expert_indices = routing.expert_indices[batch_index][utterance_index]
                for rank, expert_index_tensor in enumerate(expert_indices):
                    expert_index = int(expert_index_tensor)
                    output = self.routing_backbone.process(expert_index, utterance_hidden)
                    selected_probability = torch.softmax(routing.scores[batch_index][utterance_index], dim=-1)[expert_index]
                    identity_gate = 1.0 + (selected_probability - selected_probability.detach())
                    output = output * identity_gate
                    selected.append(SelectedExpertOutput(batch_index, utterance_index, rank, expert_index, output))
            selected_by_batch.append(selected)
        return selected_by_batch

    def encode_and_route(
        self,
        batch: DialogueBatch,
    ) -> tuple[torch.Tensor, RoutingOutput, list[list[SelectedExpertOutput]]]:
        dialogue_embeddings, dialogue_hidden = self.routing_backbone.encode(batch.input_ids, batch.attention_mask)
        routing = self.router(dialogue_hidden, batch.routing_positions)
        selected = self.route_and_process(dialogue_hidden, batch.utterance_spans, routing)
        return dialogue_embeddings, routing, selected

    def fusion_dialogue_embeddings(self, dialogue_embeddings: torch.Tensor, batch: DialogueBatch, index: int) -> torch.Tensor:
        source_length = int(batch.attention_mask[index].sum())
        return dialogue_embeddings[index, :source_length]

    def forward(self, batch: DialogueBatch) -> DiaSumMoEOutput:
        dialogue_embeddings, routing, selected = self.encode_and_route(batch)
        conditions: list[torch.Tensor] = []
        for index, expert_outputs in enumerate(selected):
            dialogue_condition = self.fusion_dialogue_embeddings(dialogue_embeddings, batch, index)
            condition = self.fusion.build_condition(dialogue_condition, expert_outputs)
            conditions.append(condition)
        loss, logits, condition_lengths = self.fusion(conditions, batch.target_ids, batch.target_attention_mask)
        return DiaSumMoEOutput(loss, logits, routing, selected, condition_lengths)

    @torch.no_grad()
    def generate(
        self,
        batch: DialogueBatch,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        eos_token_id: int | None,
        pad_token_id: int | None,
    ) -> tuple[list[torch.Tensor], RoutingOutput, list[list[SelectedExpertOutput]]]:
        dialogue_embeddings, routing, selected = self.encode_and_route(batch)
        sequences: list[torch.Tensor] = []
        for index, expert_outputs in enumerate(selected):
            dialogue_condition = self.fusion_dialogue_embeddings(dialogue_embeddings, batch, index)
            condition = self.fusion.build_condition(dialogue_condition, expert_outputs)
            sequence = self.fusion.generate(
                condition,
                max_new_tokens,
                num_beams,
                length_penalty,
                eos_token_id,
                pad_token_id,
            )
            sequences.append(sequence.squeeze(0))
        return sequences, routing, selected
