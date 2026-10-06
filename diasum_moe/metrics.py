from collections import defaultdict
from statistics import mean


def metric_text(text: str, language: str) -> str:
    normalized = " ".join(text.strip().split())
    if language.lower().startswith("zh"):
        normalized = " ".join(character for character in normalized if not character.isspace())
    return normalized


class ChineseTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return metric_text(text, "zh").split()


def overlap_metrics(predictions: list[str], references: list[str], language: str) -> dict[str, float]:
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions and references must be nonempty and aligned")
    language = language.lower()
    try:
        from rouge_score import rouge_scorer
        import sacrebleu
    except ImportError as error:
        raise RuntimeError("ROUGE and BLEU require rouge-score and sacrebleu") from error
    tokenizer = ChineseTokenizer() if language.startswith("zh") else None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=language.startswith("en"), tokenizer=tokenizer)
    scores: dict[str, list[float]] = defaultdict(list)
    normalized_predictions = [metric_text(text, language) for text in predictions]
    normalized_references = [metric_text(text, language) for text in references]
    for prediction, reference in zip(normalized_predictions, normalized_references):
        result = scorer.score(reference, prediction)
        scores["rouge1"].append(result["rouge1"].fmeasure * 100)
        scores["rouge2"].append(result["rouge2"].fmeasure * 100)
        scores["rougeL"].append(result["rougeL"].fmeasure * 100)
    bleu = sacrebleu.corpus_bleu(normalized_predictions, [normalized_references], tokenize="none").score
    return {"rouge1": mean(scores["rouge1"]), "rouge2": mean(scores["rouge2"]), "rougeL": mean(scores["rougeL"]), "bleu": bleu}


def bertscore_metric(predictions: list[str], references: list[str], language: str, model_type: str | None) -> float:
    try:
        from bert_score import score
    except ImportError as error:
        raise RuntimeError("BERTScore requires bert-score") from error
    _, _, f1 = score(predictions, references, lang=language, model_type=model_type, verbose=False)
    return float(f1.mean() * 100)


def moverscore_metric(predictions: list[str], references: list[str]) -> float:
    try:
        from moverscore_v2 import get_idf_dict, word_mover_score
    except ImportError as error:
        raise RuntimeError("MoverScore requires moverscore-v2") from error
    idf_reference = get_idf_dict(references)
    idf_prediction = get_idf_dict(predictions)
    values = word_mover_score(
        references,
        predictions,
        idf_reference,
        idf_prediction,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )
    return mean(float(value) for value in values) * 100




