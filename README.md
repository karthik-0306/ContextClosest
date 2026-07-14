# ContextCloset — Multimodal Fashion & Context Retrieval

ContextCloset is a hybrid image retrieval system designed to solve the compositionality problem in standard CLIP models (e.g., confusing "red shirt, blue pants" with "blue shirt, red pants"). 

It combines **Marqo-FashionCLIP** for raw vector similarity with **Gemini 2.0 Flash / Llama 3.2 90B Vision** for structured attribute tagging and re-ranking.

## Setup

1. **Environment:** Use a conda environment.
```bash
conda activate ai_env
pip install -r requirements.txt
```

2. **API Keys:**
Set your API keys for the tag extraction:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_gemini_key"
$env:GROQ_API_KEY="your_groq_key"
```

## Running the Pipeline

### Part A: Indexing
The indexer downloads the first 800 images from the `fashionpedia` dataset, embeds them, extracts attributes, and stores them in ChromaDB.
```bash
python -m indexer.build_index
```

### Part B: Evaluation
The evaluation script runs 5 required queries through the retriever and saves matplotlib grids to the `results/` directory.
```bash
python evaluate.py
```

## Architecture

- **Indexer (`indexer/`)**: Handles dataset ingestion, image embedding (`embed.py`), tagging (`tag.py`), and ChromaDB storage (`build_index.py`).
- **Retriever (`retriever/`)**: Parses queries into JSON (`query_parser.py`) and executes a two-stage hybrid search (`search.py`) that scores candidates on vector similarity and attribute match.
