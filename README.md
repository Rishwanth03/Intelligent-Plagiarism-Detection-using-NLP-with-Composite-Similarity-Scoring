# Automated Plagiarism Detection System

A production-ready full-stack plagiarism detection platform with:

- Python NLP engine (TF-IDF + lexical + n-gram)
- Flask modular REST API backend
- React dashboard frontend
- Comprehensive unittest suite (40+ tests)
- Docker-ready deployment layout

## System Architecture

- `backend/app/nlp/plagiarism_detector.py`: core NLP logic
- `backend/app/routes`: API route blueprints
- `backend/app/services`: business logic and orchestration
- `backend/app/models`: dataclasses for documents/results
- `frontend/src`: React pages/components/api integration

## Scoring Logic

Composite score:

```
0.5 * cosine_similarity + 0.3 * ngram_similarity + 0.2 * lexical_overlap
```

Classification thresholds:

- `<10%`: No Plagiarism
- `10-30%`: Minor
- `30-60%`: Moderate
- `>60%`: Severe

## Backend Setup

1. Create virtual environment and install dependencies:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Download NLTK resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

3. Run API server:

```bash
python run.py
```

The backend automatically trains the TF-IDF model from:

`train_snli.txt/train_snli.txt`

Optional override:

```bash
set TRAIN_DATA_PATH=C:\path\to\your\train_snli.txt
set TRAIN_MAX_LINES=20000
python run.py
```

Server URL: `http://localhost:5000`

## Rewrite LLM Providers

The rewrite engine supports multiple providers. Configure these in your `.env` file:

- `REWRITE_AI_PROVIDER=none` for heuristic-only rewrite (default)
- `REWRITE_AI_PROVIDER=openai`
- `REWRITE_AI_PROVIDER=ollama`
- `REWRITE_AI_PROVIDER=anthropic`
- `REWRITE_AI_PROVIDER=gemini`

Provider environment variables:

- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_REWRITE_MODEL`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_REWRITE_MODEL`
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_REWRITE_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_REWRITE_MODEL`

Example (OpenAI):

```bash
set REWRITE_AI_PROVIDER=openai
set OPENAI_API_KEY=your_key
set OPENAI_REWRITE_MODEL=gpt-4o-mini
```

Example (Ollama local):

```bash
set REWRITE_AI_PROVIDER=ollama
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_REWRITE_MODEL=llama3.1:8b
```

If provider configuration is missing or unreachable, the service falls back to the built-in heuristic rewrite pipeline.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard URL: `http://localhost:5173`

To change backend URL, set `VITE_API_BASE_URL`.

Bundle budget checks (frontend):

```bash
cd frontend
npm run build:ci
```

Optional environment overrides:

```bash
set MAX_ENTRY_BUNDLE_KB=260
set MAX_CHUNK_BUNDLE_KB=420
```

A GitHub Actions workflow at `.github/workflows/frontend-ci.yml` runs this check on frontend changes.

Source/Target upload in the dashboard supports:

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.doc` (best-effort legacy parsing)

## API Examples

### Create document

```http
POST /api/documents
Content-Type: application/json

{
  "text": "This is my document content"
}
```

### Composite similarity

```http
POST /api/similarity/composite
Content-Type: application/json

{
  "text1": "Machine learning is useful",
  "text2": "Machine learning is very useful"
}
```

### Detect plagiarism

```http
POST /api/detect
Content-Type: application/json

{
  "source_text": "original sentence",
  "target_text": "copied sentence"
}
```

### Reports

- `GET /api/report/pairwise`
- `GET /api/report/statistics`

## Testing

```bash
cd backend
python -m unittest discover -s tests -v
```

## Performance and Scalability Notes

- Vectorizer limited to 5000 features with unigram/bigram support
- Lightweight in-memory document store for low-latency comparisons
- Batch-oriented pairwise reporting for large document sets
- Optional persistence can be enabled in service configuration
- Designed to scale to 1000+ documents for analytical workloads

## Deployment

Docker and compose files are provided:

```bash
docker compose up --build
```

## Extensibility Roadmap

- Transformer-based semantic similarity (BERT/SBERT)
- Multilingual tokenization + language detection
- Authorship attribution models
- External plagiarism corpus and search integration
