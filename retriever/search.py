import os
import sys
import json
import chromadb
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indexer.embed import FashionCLIPEmbedder
from retriever.query_parser import QueryParser

class FashionRetriever:
    def __init__(self):
        self.embedder = FashionCLIPEmbedder(config.CLIP_MODEL_NAME)
        self.parser = QueryParser(config.GEMINI_MODEL, config.GROQ_TEXT_MODEL)
        
        client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
        self.collection = client.get_collection(name="fashion")
        
    def compute_attribute_score(self, query_tags: dict, doc_tags: dict) -> float:
        """
        Compare query attributes with document attributes to calculate a score.
        Score is based on matching items, colors, and setting.
        """
        score = 0.0
        max_score = 1e-5 # prevent division by zero
        
        # Setting match
        q_setting = query_tags.get("setting", "").lower()
        d_setting = doc_tags.get("setting", "").lower()
        if q_setting and q_setting != "unknown":
            max_score += 1.0
            # Simple keyword overlap for setting
            q_words = set(q_setting.split())
            d_words = set(d_setting.split())
            if q_words.intersection(d_words):
                score += 1.0
                
        # Garments match
        q_garments = query_tags.get("garments", [])
        d_garments = doc_tags.get("garments", [])
        
        for q_g in q_garments:
            max_score += 2.0 # 1 for item, 1 for color
            q_item = q_g.get("item", "").lower()
            q_color = q_g.get("color", "").lower()
            
            best_match = 0.0
            for d_g in d_garments:
                d_item = d_g.get("item", "").lower()
                d_color = d_g.get("color", "").lower()
                
                current_match = 0.0
                # Item match
                if q_item and d_item and (q_item in d_item or d_item in q_item):
                    current_match += 1.0
                # Color match
                if q_color and d_color and (q_color in d_color or d_color in q_color):
                    current_match += 1.0
                    
                best_match = max(best_match, current_match)
            score += best_match
            
        return score / max_score

    def search(self, query: str, top_k: int = 5, recall_k: int = 20, alpha: float = 0.6):
        """
        Hybrid search:
        1. Recall: get `recall_k` candidates using vector similarity.
        2. Precision: re-rank using attribute match score.
        alpha is the weight for vector similarity. (1 - alpha) for attribute match.
        """
        print(f"Parsing query: '{query}'")
        query_tags = self.parser.parse_query(query)
        print(f"Parsed Tags: {json.dumps(query_tags)}")
        
        query_embedding = self.embedder.embed_texts([query])[0]
        
        # Stage 1: Vector Search (Recall)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=recall_k,
            include=["metadatas", "documents", "distances"]
        )
        
        if not results['ids'][0]:
            return []
            
        candidates = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            # We used cosine space in chroma. Distance is cosine distance. Sim = 1 - dist.
            vector_dist = results['distances'][0][i]
            vector_sim = 1.0 - vector_dist
            
            meta = results['metadatas'][0][i]
            doc_path = results['documents'][0][i]
            
            doc_tags = json.loads(meta['tags'])
            
            # Stage 2: Attribute Score
            attr_score = self.compute_attribute_score(query_tags, doc_tags)
            
            # Final Score
            final_score = alpha * vector_sim + (1 - alpha) * attr_score
            
            candidates.append({
                "id": doc_id,
                "path": doc_path,
                "vector_sim": vector_sim,
                "attr_score": attr_score,
                "final_score": final_score,
                "tags": doc_tags
            })
            
        # Re-rank by final_score descending
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_k]
