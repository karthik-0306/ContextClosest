# ContextCloset — Multimodal Fashion & Context Retrieval

ContextCloset is a hybrid image retrieval system designed to solve the compositionality problem in standard CLIP models (e.g., confusing "red shirt, blue pants" with "blue shirt, red pants"). 

It combines **Marqo-FashionCLIP** for raw vector similarity with **Vision-Language Models (VLMs)** for structured attribute tagging and re-ranking.

## Setup

1. **Install Dependencies:**
   Install the required Python packages (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys:**
   Create a `.env` file in the root of the project and add your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_key_here
   GROQ_API_KEY_1=your_groq_key_here
   ```
   *(Note: The system supports automatic key rotation if you provide multiple Groq keys as `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, etc.)*

3. **Data:**
   Ensure your images are placed in the `data/test/` directory.

## Running the Pipeline

### Part A: Indexing
The indexer scans the `data/test/` directory, embeds the images using FashionCLIP, extracts structured attributes using a VLM, and stores them in ChromaDB.
*(Note: Indexing supports resuming if interrupted).*
```bash
python -m indexer.build_index
```

### Part B: Evaluation
The evaluation script runs 5 specific semantic/compositional queries through the retriever and saves the visual result grids to the `results/` directory.
```bash
python evaluate.py
```

## Architecture

- **Indexer (`indexer/`)**: Handles dataset ingestion, image embedding (`embed.py`), tagging (`tag.py`), and ChromaDB storage (`build_index.py`).
- **Retriever (`retriever/`)**: Parses queries into structured JSON (`query_parser.py`) and executes a two-stage hybrid search (`search.py`) that scores candidates on vector similarity and exact attribute matches.
