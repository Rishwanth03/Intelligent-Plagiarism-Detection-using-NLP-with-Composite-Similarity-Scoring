# Methodology and Evaluation Report (Optional)

## Abstract

This system combines lexical and statistical NLP methods for interpretable plagiarism detection in educational and editorial workflows.

## Methodology

- Representation: TF-IDF unigram/bigram vectors
- Structural overlap: Jaccard on 2-gram/3-gram sets
- Lexical overlap: word-set containment ratio
- Aggregation: weighted composite score emphasizing semantic closeness through cosine similarity

## Metrics

The dashboard exposes:

- Precision
- Recall
- Composite similarity trends
- Pairwise comparison distributions

## Interpretation

- High cosine + high n-gram indicates likely copy with minor paraphrase
- Moderate cosine + lexical overlap indicates thematic resemblance
- Low scores across all methods indicates independent writing

## Limitations

- Semantic paraphrase can bypass lexical methods
- Non-English text quality depends on tokenization resources
- No external corpus lookup in baseline mode

## Planned Experimental Expansion

- Benchmark against transformer-based sentence embeddings
- Evaluate on multilingual plagiarism datasets
- Measure throughput/latency under 1000+ document workloads
