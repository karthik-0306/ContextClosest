# Multimodal Fashion & Context Retrieval
**Glance ML Internship Assignment Submission**

## Part I: Submission Deliverables

### 1. Approaches Considered

When building a multimodal search engine for fashion and context, several architectural approaches can be evaluated, each with distinct tradeoffs:

*   **Approach A: Pure Zero-Shot CLIP (Baseline)**
    *   **How it works:** Compute cosine similarity between the natural language query and raw images using a pre-trained CLIP model.
    *   **What's good:** Extremely simple and fast to implement. Requires no metadata extraction.
    *   **Tradeoffs:** CLIP is notoriously bad at **compositionality**. It struggles to distinguish "red shirt with blue pants" from "blue shirt with red pants." It behaves like a bag-of-concepts, which hurts precision for fine-grained fashion queries.
    *   **When to use:** For high-level, general "vibe" search where precise attribute matching is not critical.

*   **Approach B: Dense Object Detection + Knowledge Graphs**
    *   **How it works:** Run object detection (e.g., YOLO) to crop garments, classify them, and build a searchable SQL/Graph database.
    *   **What's good:** Flawless compositionality and exact attribute matching.
    *   **Tradeoffs:** Highly rigid. Cannot handle abstract queries like "casual weekend vibe" well. Very compute-intensive to index.
    *   **When to use:** In strict e-commerce environments requiring exact SKU retrieval.

*   **Approach C: Hybrid Search (The Chosen Approach)**
    *   **How it works:** Combines the semantic "vibe" understanding of CLIP with the compositional precision of Vision-Language Models (VLMs).
    *   **What's good:** Solves compositionality by using structured metadata, while retaining zero-shot semantic capabilities.
    *   **Tradeoffs:** Indexing requires an LLM/VLM, making the ingestion phase slower and slightly more expensive. 
    *   **When to use:** When you need a balance of precise garment matching and abstract context understanding.

---

### 2. Short Write-up on Chosen Approach

For this assignment, I implemented the **Hybrid Search Architecture**.

*   **The Indexer (Feature Extraction & Storage):** 
    We use `Marqo/marqo-fashionCLIP` to generate a dense semantic vector for each image. Simultaneously, the image is passed through a Vision-Language Model (Llama-3-Vision/Qwen via Groq) to extract structured JSON metadata (identifying the `setting` and a list of specific `garments` with colors). Both the vector and the JSON metadata are stored efficiently in **ChromaDB**.
*   **The Retriever (Search Logic & Context Awareness):**
    When a complex multi-attribute query is received (e.g., "A red tie and a white shirt in a formal setting"), the system works in two stages:
    1.  **Recall (Vector Search):** The query is embedded via FashionCLIP to retrieve the top $K$ visually and semantically similar images from ChromaDB.
    2.  **Precision (Re-ranking):** A fast LLM parses the user's text query into the same JSON schema used during indexing. We compute an `Attribute Match Score` based on exact intersections of colors, clothing types, and locations. The final retrieved list is a weighted blend (e.g., 60% semantic similarity + 40% exact attribute match).

---

### 3. Codebase (GitHub) Link

**Repository:** `[INSERT YOUR GITHUB LINK HERE]`
*(Note: The `results/` folder in the repository contains the visual outputs of the 5 evaluation queries demonstrating the system's accuracy).*

---

### 4. Approaches for Future Work

*   **a. How to extend this solution for adding locations (cities, places) and weather:**
    Because the architecture uses LLM-driven metadata extraction, extending this is trivial. We would simply add `weather` and `city` keys to the JSON schema prompt during indexing. During retrieval, instead of relying purely on soft hybrid scoring, we can pass these extracted entities as **hard metadata filters** directly into ChromaDB (e.g., `where={"weather": "snowing"}`). This guarantees that only snowy images are returned, vastly improving speed and accuracy.
*   **b. How to improve precision:**
    1.  **Cross-Encoder Re-ranking:** Replace the heuristic attribute-matching logic with a lightweight Vision-Language Cross-Encoder that scores the query and image pairs jointly.
    2.  **Bounding Box Grounding:** Have the VLM extract bounding boxes for garments during indexing. During retrieval, we can crop the specific garment and verify its color/texture independently of the background.

---
---

## Part II: Assignment Assessment Criteria ("What We Are Looking For")

### 1. Thoughtful Solution
*   **Shortcomings & Solutions:** The main shortcoming of this hybrid approach is the reliance on a VLM during indexing, which can hallucinate colors or misinterpret lighting. To address this, we resize images before VLM inference (to save tokens) and use temperature=0.0 to force deterministic outputs. If the VLM fails, the system gracefully falls back to pure FashionCLIP vector search.
*   **Performance on Fashion Queries:** It excels. By separating the "semantic vibe" (CLIP) from "compositional facts" (VLM attributes), queries like *"a red tie and a white shirt"* don't get confused with *"a white tie and a red shirt"*.

### 2. Modular Code
**Yes, logic is strictly separated from data.** 
*   Data (images) and DB state (`chroma_db/`) are ignored in version control.
*   The codebase is split into distinct modules: `/indexer/` (handling ingestion) and `/retriever/` (handling queries).
*   Configuration and paths are centrally managed in `config.py`.

### 3. Scalability
**Yes, this logic scales to 1 million images.** 
We chose ChromaDB specifically because it utilizes HNSW (Hierarchical Navigable Small World) graphs for vector storage. HNSW search time scales logarithmically $O(\log N)$, meaning searching 1 million images takes fractions of a millisecond longer than searching 1,000. Furthermore, because we do the expensive LLM attribute extraction only once during *indexing*, the *retrieval* step remains incredibly fast and lightweight.

### 4. Zero-Shot Capability
**Highly Capable.** 
Because we use FashionCLIP for our base recall layer, the system has seen hundreds of millions of image-text pairs during its pre-training. If a user queries a concept that wasn't explicitly in the training labels (e.g., "cyberpunk streetwear"), CLIP's latent space still understands the semantic proximity of those words to visual neon/urban aesthetics. The system successfully returns relevant images even without explicit prior tagging.
