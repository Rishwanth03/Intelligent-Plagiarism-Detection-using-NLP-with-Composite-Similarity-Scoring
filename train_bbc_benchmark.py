import random
import os
from pathlib import Path

from app.nlp.plagiarism_detector import PlagiarismDetector


def safe_read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_bbc_pairs(dataset_root: Path) -> list[tuple[Path, Path, str]]:
    articles_root = dataset_root / "News Articles"
    summaries_root = dataset_root / "Summaries"

    pairs: list[tuple[Path, Path, str]] = []
    categories = ["business", "entertainment", "politics", "sport", "tech"]

    for category in categories:
        article_cat = articles_root / category
        summary_cat = summaries_root / category
        if not article_cat.exists() or not summary_cat.exists():
            continue

        article_files = sorted(article_cat.glob("*.txt"))
        for article in article_files:
            summary = summary_cat / article.name
            if summary.exists():
                pairs.append((article, summary, category))

    return pairs


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bbc_root = root / "BBC News Summary"
    snli_csv = root / "train_snli.txt" / "plagiarism_dataset_120.csv"

    if not bbc_root.exists():
        raise FileNotFoundError(f"BBC dataset not found: {bbc_root}")

    detector = PlagiarismDetector(max_features=5000)
    if snli_csv.exists():
        detector.train_from_file(str(snli_csv), max_lines=None)

    pairs = collect_bbc_pairs(bbc_root)
    if not pairs:
        raise RuntimeError("No BBC article-summary pairs found")

    max_pairs_raw = os.getenv("BBC_MAX_PAIRS", "300")
    try:
        max_pairs = max(1, int(max_pairs_raw))
    except ValueError:
        max_pairs = 300
    pairs = pairs[:max_pairs]

    random.seed(42)

    by_category: dict[str, list[tuple[Path, Path, str]]] = {}
    for pair in pairs:
        by_category.setdefault(pair[2], []).append(pair)

    y_true: list[int] = []
    pred_tuned: list[int] = []
    pred_composite: list[int] = []
    tuned_scores: list[float] = []
    composite_scores: list[float] = []

    for article_path, summary_path, category in pairs:
        article_text = safe_read_text(article_path)
        summary_text = safe_read_text(summary_path)

        cosine = detector.cosine_similarity(article_text, summary_text)
        ngram = detector.ngram_similarity(article_text, summary_text)
        lexical = detector.lexical_overlap(article_text, summary_text)
        semantic = detector.semantic_similarity(article_text, summary_text)
        composite = (0.5 * cosine) + (0.3 * ngram) + (0.2 * lexical)
        tuned = detector.tuned_score(cosine, ngram, lexical, semantic)

        y_true.append(1)
        pred_tuned.append(1 if tuned >= detector.tuned_threshold else 0)
        pred_composite.append(1 if composite >= 0.5 else 0)
        tuned_scores.append(tuned)
        composite_scores.append(composite)

        alternatives = [p for p in by_category.get(category, []) if p[1] != summary_path]
        if alternatives:
            neg_summary_path = random.choice(alternatives)[1]
            neg_summary_text = safe_read_text(neg_summary_path)
            cosine = detector.cosine_similarity(article_text, neg_summary_text)
            ngram = detector.ngram_similarity(article_text, neg_summary_text)
            lexical = detector.lexical_overlap(article_text, neg_summary_text)
            semantic = detector.semantic_similarity(article_text, neg_summary_text)
            composite = (0.5 * cosine) + (0.3 * ngram) + (0.2 * lexical)
            tuned = detector.tuned_score(cosine, ngram, lexical, semantic)

            y_true.append(0)
            pred_tuned.append(1 if tuned >= detector.tuned_threshold else 0)
            pred_composite.append(1 if composite >= 0.5 else 0)
            tuned_scores.append(tuned)
            composite_scores.append(composite)

    acc_tuned = accuracy(y_true, pred_tuned)
    p_tuned, r_tuned, f1_tuned = precision_recall_f1(y_true, pred_tuned)

    acc_comp = accuracy(y_true, pred_composite)
    p_comp, r_comp, f1_comp = precision_recall_f1(y_true, pred_composite)

    print("BBC BENCHMARK SUMMARY")
    print(f"dataset_root={bbc_root}")
    print(f"pairs={len(pairs)}")
    print(f"max_pairs={max_pairs}")
    print(f"evaluated_examples={len(y_true)}")
    print()
    print("TUNED MODEL METRICS")
    print(f"accuracy={acc_tuned:.4f}")
    print(f"precision={p_tuned:.4f}")
    print(f"recall={r_tuned:.4f}")
    print(f"f1={f1_tuned:.4f}")
    print()
    print("BASELINE COMPOSITE METRICS (threshold=0.5)")
    print(f"accuracy={acc_comp:.4f}")
    print(f"precision={p_comp:.4f}")
    print(f"recall={r_comp:.4f}")
    print(f"f1={f1_comp:.4f}")
    print()
    print("SCORE AVERAGES")
    print(f"avg_tuned_score={sum(tuned_scores) / len(tuned_scores):.4f}")
    print(f"avg_composite_score={sum(composite_scores) / len(composite_scores):.4f}")
    print()
    print("SUGGESTED ENV CONFIG")
    print("set TUNED_PROFILE=bbc")
    print("set TUNED_WEIGHTS_BBC=0.50,0.30,0.20,0.00")
    print("set TUNED_THRESHOLD_BBC=0.50")


if __name__ == "__main__":
    main()
