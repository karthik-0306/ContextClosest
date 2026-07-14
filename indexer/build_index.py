import os
import sys
import glob
import json
import chromadb
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indexer.embed import FashionCLIPEmbedder
from indexer.tag import AttributeTagger

NUM_IMAGES = 800
BATCH_SIZE = 32

def build_index():
    test_dir = os.path.join(config.DATA_DIR, "test")
    print(f"Scanning for local images in {test_dir}...")
    image_paths = sorted(
        glob.glob(os.path.join(test_dir, "*.jpg")) +
        glob.glob(os.path.join(test_dir, "*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {test_dir}.")

    selected_paths = image_paths[:NUM_IMAGES]
    print(f"Found {len(selected_paths)} images for indexing.")

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="fashion", metadata={"hnsw:space": "cosine"})

    # Resume: skip images that are already indexed
    existing = set(collection.get(include=[])["ids"])
    pending_paths = [
        p for p in selected_paths
        if os.path.splitext(os.path.basename(p))[0] not in existing
    ]

    if not pending_paths:
        print(f"All {len(selected_paths)} images already indexed. Nothing to do.")
        return

    print(f"Resuming: {len(existing)} already done, {len(pending_paths)} remaining.")

    print("Initializing embedder and tagger...")
    embedder = FashionCLIPEmbedder(config.CLIP_MODEL_NAME)
    tagger = AttributeTagger(config.GEMINI_MODEL, config.GROQ_VISION_MODEL)

    total_batches = (len(pending_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(total_batches), desc="Indexing batches"):
        start = batch_idx * BATCH_SIZE
        batch_paths = pending_paths[start : start + BATCH_SIZE]

        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if not images:
            continue

        embeddings = embedder.embed_images(images)

        ids = []
        metadatas = []
        for j, img in enumerate(images):
            base_id = os.path.splitext(os.path.basename(valid_paths[j]))[0]
            tags = tagger.tag_image(img)
            ids.append(base_id)
            metadatas.append({
                "tags": json.dumps(tags),
                "setting": tags.get("setting", "")
            })

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=valid_paths
        )

    total_count = collection.count()
    print(f"Indexing complete. ChromaDB now contains {total_count} images.")

if __name__ == "__main__":
    build_index()
