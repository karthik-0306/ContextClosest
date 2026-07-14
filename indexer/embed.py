import torch
import open_clip
from typing import List
from PIL import Image
import numpy as np

class FashionCLIPEmbedder:
    def __init__(self, model_name: str = "hf-hub:Marqo/marqo-fashionCLIP"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed a batch of images and return L2-normalized vectors.
        """
        processed_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_images)
            # L2 Normalization
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
        return image_features.cpu().numpy()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts and return L2-normalized vectors.
        """
        text_tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # L2 Normalization
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
        return text_features.cpu().numpy()
