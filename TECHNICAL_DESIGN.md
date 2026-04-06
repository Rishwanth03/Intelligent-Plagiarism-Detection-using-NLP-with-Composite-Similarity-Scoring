# Technical Design Document

## 1. Objectives

Build a modular plagiarism detection platform with maintainable architecture, robust REST APIs, and a modern analytics dashboard.

## 2. Architecture

### Backend (Flask)

- Routes layer: HTTP request parsing/validation and response mapping
- Service layer: business logic orchestration
- NLP engine: text preprocessing + scoring algorithms
- Models layer: document/result representations
- Utils layer: shared validators and exceptions

### Frontend (React)

- Page-level route components for each workflow
- Reusable UI components (`UploadForm`, `ResultCard`, `Charts`, `Navbar`)
- Centralized API adapter (`src/api/api.js`)

## 3. Detection Algorithm

### 3.1 Preprocessing

1. Lowercase conversion
2. URL and email removal
3. Special character stripping
4. Tokenization via NLTK (`word_tokenize`)
5. Stopword removal via NLTK stopword corpus
6. Lemmatization with WordNet lemmatizer

### 3.2 Similarity Measures

1. Cosine similarity over TF-IDF vectors
2. N-gram Jaccard similarity over bigrams + trigrams
3. Lexical overlap ratio on token sets

### 3.3 Composite

\[
S = 0.5C + 0.3N + 0.2L
\]

Where:

- \(C\) = cosine similarity
- \(N\) = n-gram Jaccard
- \(L\) = lexical overlap

### 3.4 Classification

- No Plagiarism: \(S < 0.10\)
- Minor: \(0.10 \leq S \leq 0.30\)
- Moderate: \(0.30 < S \leq 0.60\)
- Severe: \(S > 0.60\)

## 4. Complexity

- Preprocessing: \(O(n)\)
- TF-IDF transform: approximately \(O(n \cdot d)\)
- Cosine similarity: \(O(d)\)
- N-gram set construction: \(O(n)\)
- Jaccard/overlap set ops: \(O(n)\) average

## 5. Reliability and Validation

- Strict JSON input validation for required text fields and IDs
- Graceful handling for empty payloads and not-found resources
- API-level and NLP-level unit tests (40+)

## 6. Deployment and Operations

- Dockerized backend/frontend services
- Environment variable based configuration
- CORS enabled for browser-based clients

## 7. Future Enhancements

- BERT/SBERT embedding similarity
- Distributed vector index for very large corpora
- Authentication/authorization and tenant isolation
- Async batch pipeline and queue-based processing
