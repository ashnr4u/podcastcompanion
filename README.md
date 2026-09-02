# Fermi Podcast Companion

A conversational RAG application for exploring the supplied Fermi podcast episodes. Answers are grounded in transcript evidence and include timestamps for verification.

## Product Note

**User:** Physics learners who want to explore podcast content conversationally.

**Problem:** Podcasts are linear, making it difficult to quickly find specific explanations or ask follow-up questions.

**Solution:** A RAG system that answers questions using transcript evidence and timestamps.

## Setup

From the project root:

```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key to `backend/.env`:

```env
GROQ_API_KEY=your_api_key
```

## Run

Start the backend:

```bash
cd backend
python src/main.py
```

Start the frontend in a separate terminal:

```bash
cd frontend
python app.py
```

## Evaluation

The repository contains **3 evaluation case sets** with runnable evaluation scripts and saved results.

From the project root, run:

```bash
cd backend
python evaluation/run_eval.py
```

The evaluator runs the cases against the product and reports:

* Source coverage
* Evidence Recall@10
* Precision@10
* Latency
* Category-level metrics

Raw outputs and retrieved evidence are saved under:

```text
backend/evaluation/
```

See [`EVAL.md`](EVAL.md) for the complete evaluation methodology, results, and failure analysis.

## Project Structure

```text
.
├── backend/
│   ├── audios/             # Supplied podcast audio
│   ├── data/               # Transcripts / knowledge-base artifacts
│   ├── evaluation/         # Evaluation cases, runner, and results
│   ├── src/                # Backend and RAG pipeline
│   ├── architecture.md     # System architecture
│   ├── notes.md
│   └── requirements.txt
│
├── frontend/
│   └── app.py              # Chat interface
│
├── EVAL.md                 # Evaluation methodology and results
├── retrievalreport.md      # Retrieval experiments
└── .gitignore
```
