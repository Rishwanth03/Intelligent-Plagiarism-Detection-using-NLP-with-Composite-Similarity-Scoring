import csv
import os
from statistics import mean

from app.nlp.plagiarism_detector import PlagiarismDetector


def load_labeled_pairs(csv_path: str):
    pairs = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            para1 = (row.get("para1") or "").strip()
            para2 = (row.get("para2") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not para1 or not para2 or label_raw not in {"0", "1"}:
                continue
            pairs.append((para1, para2, int(label_raw)))
    return pairs


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def weighted_score(cosine, ngram, lexical, semantic, weights):
    return (
        weights[0] * cosine
        + weights[1] * ngram
        + weights[2] * lexical
        + weights[3] * semantic
    )


def tune_weights_and_threshold(cosine_scores, ngram_scores, lexical_scores, semantic_scores, y_true):
    # Coarse grid search over normalized weights with step=0.1 and threshold sweep.
    best = {
        "accuracy": -1.0,
        "weights": (0.5, 0.3, 0.2, 0.0),
        "threshold": 0.5,
    }

    weight_values = [i / 10 for i in range(0, 11)]
    thresholds = [i / 100 for i in range(20, 81)]

    for wc in weight_values:
        for wn in weight_values:
            for wl in weight_values:
                ws = 1.0 - (wc + wn + wl)
                if ws < 0:
                    continue
                weights = (wc, wn, wl, ws)

                combined_scores = [
                    weighted_score(c, n, l, s, weights)
                    for c, n, l, s in zip(
                        cosine_scores,
                        ngram_scores,
                        lexical_scores,
                        semantic_scores,
                    )
                ]

                for threshold in thresholds:
                    y_pred = [1 if score >= threshold else 0 for score in combined_scores]
                    acc = accuracy(y_true, y_pred)
                    if acc > best["accuracy"]:
                        best = {
                            "accuracy": acc,
                            "weights": weights,
                            "threshold": threshold,
                        }
    return best


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "train_snli.txt", "plagiarism_dataset_120.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    detector = PlagiarismDetector(max_features=5000)
    trained_size = detector.train_from_file(csv_path, max_lines=None)

    pairs = load_labeled_pairs(csv_path)
    y_true = [label for _, _, label in pairs]

    cosine_scores = [detector.cosine_similarity(a, b) for a, b, _ in pairs]
    ngram_scores = [detector.ngram_similarity(a, b) for a, b, _ in pairs]
    lexical_scores = [detector.lexical_overlap(a, b) for a, b, _ in pairs]
    semantic_scores = [detector.semantic_similarity(a, b) for a, b, _ in pairs]
    composite_scores = [detector.composite_score(a, b) for a, b, _ in pairs]

    threshold = 0.5
    y_pred_composite = [1 if score >= threshold else 0 for score in composite_scores]
    y_pred_semantic = [1 if score >= threshold else 0 for score in semantic_scores]

    tuned = tune_weights_and_threshold(
        cosine_scores,
        ngram_scores,
        lexical_scores,
        semantic_scores,
        y_true,
    )

    print("TRAINING SUMMARY")
    print(f"dataset_path={csv_path}")
    print(f"pairs={len(pairs)}")
    print(f"trained_corpus_size={trained_size}")
    print("\nMODEL METRICS (using threshold=0.5 for binary label check)")
    print(f"composite_accuracy={accuracy(y_true, y_pred_composite):.4f}")
    print(f"semantic_accuracy={accuracy(y_true, y_pred_semantic):.4f}")
    print(
        "tuned_accuracy="
        f"{tuned['accuracy']:.4f} "
        f"weights(cosine,ngram,lexical,semantic)={tuned['weights']} "
        f"threshold={tuned['threshold']:.2f}"
    )
    print("\nFEATURE AVERAGES")
    print(f"avg_cosine={mean(cosine_scores):.4f}")
    print(f"avg_ngram={mean(ngram_scores):.4f}")
    print(f"avg_lexical={mean(lexical_scores):.4f}")
    print(f"avg_semantic={mean(semantic_scores):.4f}")
    print(f"avg_composite={mean(composite_scores):.4f}")

    print("\nSUGGESTED ENV CONFIG")
    wc, wn, wl, ws = tuned["weights"]
    print("set TUNED_PROFILE=pairwise")
    print(f"set TUNED_WEIGHTS_PAIRWISE={wc:.2f},{wn:.2f},{wl:.2f},{ws:.2f}")
    print(f"set TUNED_THRESHOLD_PAIRWISE={tuned['threshold']:.2f}")


if __name__ == "__main__":
    main()
