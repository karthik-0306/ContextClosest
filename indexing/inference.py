"""
indexing/inference.py

Handles loading the fine-tuned SigLIP 2 model (dense) and the SPLADE model (sparse)
for embedding generation. Runs the same PyTorch path locally and in the cloud
(Hugging Face Spaces).
"""

import logging
import os
from pathlib import Path

# fastembed natively uses ONNX under the hood and is lightweight
from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Wraps the fine-tuned SigLIP 2 model (dense) and SPLADE model (sparse)
    for embedding generation.
    """

    def __init__(self, base_model_name: str, lora_adapter_path: str) -> None:
        """
        Initializes the dense and sparse embedding models.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        logger.info("Initializing InferenceEngine (PyTorch).")
        # Cap CPU threads.  Using every logical core oversubscribes hybrid laptop
        # CPUs (P+E cores) and makes single-query inference ~10x slower (measured
        # 4.2 s at 12 threads vs 0.5 s at 4 on an i5-1335U); it also lets the API
        # and the Kafka streaming indexer share one machine.  Override with
        # TORCH_NUM_THREADS.
        usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        num_threads = int(os.getenv("TORCH_NUM_THREADS", min(4, usable)))
        torch.set_num_threads(num_threads)
        logger.info("PyTorch using %d CPU threads.", num_threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        base_model = AutoModel.from_pretrained(base_model_name)

        if Path(lora_adapter_path).exists():
            from peft import PeftModel

            logger.info("Applying LoRA adapter from %s", lora_adapter_path)
            self.dense_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            logger.warning(
                "LoRA adapter not found at %s. Using base model.", lora_adapter_path
            )
            self.dense_model = base_model

        self.dense_model.to(self.device)
        self.dense_model.eval()

        # SPLADE Sparse Model (FastEmbed / ONNX under the hood)
        logger.info("Loading SPLADE sparse embedding model...")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    def embed_text_dense(self, texts: list[str]) -> list[list[float]]:
        """
        Generates dense SigLIP 2 embeddings for text queries.
        """
        import torch

        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.dense_model.base_model.get_text_features(**inputs)

            if hasattr(text_embeddings, "text_embeds"):
                text_embeddings = text_embeddings.text_embeds
            elif hasattr(text_embeddings, "pooler_output"):
                text_embeddings = text_embeddings.pooler_output
            elif not isinstance(text_embeddings, torch.Tensor):
                text_embeddings = text_embeddings[0]

            text_embeddings = text_embeddings / text_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )
            return text_embeddings.cpu().numpy().tolist()

    def embed_text_sparse(self, texts: list[str]) -> list[dict]:
        """
        Generates sparse SPLADE embeddings for text.
        """
        embeddings_iter = self.sparse_model.embed(texts)
        results = []
        for emb in embeddings_iter:
            results.append(
                {
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                }
            )
        return results

    def embed_images_dense(self, images: list) -> list[list[float]]:
        """
        Generates dense SigLIP 2 embeddings for images (used by the offline indexer).
        """
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeddings = self.dense_model.base_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

            if hasattr(image_embeddings, "image_embeds"):
                image_embeddings = image_embeddings.image_embeds
            elif hasattr(image_embeddings, "pooler_output"):
                image_embeddings = image_embeddings.pooler_output
            elif not isinstance(image_embeddings, torch.Tensor):
                image_embeddings = image_embeddings[0]

            image_embeddings = image_embeddings / image_embeddings.norm(
                p=2, dim=-1, keepdim=True
            )
            return image_embeddings.cpu().numpy().tolist()
