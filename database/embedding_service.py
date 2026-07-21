from __future__ import annotations

import os
from typing import List

import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    logger.warning("FAISS not found. Falling back to manual cosine similarity for all searches.")
    HAS_FAISS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers not found. Embedding service will fail.")
    HAS_SENTENCE_TRANSFORMERS = False

from config import get_agent_config


# ── Global Model Singleton ────────────────────────────────────────────────────
_shared_model: SentenceTransformer | None = None

def get_shared_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer | None:
    """Lazy-load the SentenceTransformer model once and share it."""
    global _shared_model
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.error("❌ sentence-transformers library is not installed. Embedding service cannot start.")
        return None
        
    if _shared_model is None:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🚀 Loading local SentenceTransformer model: {model_name} (Attempt {attempt + 1}/{max_retries + 1})...")
                # Use a timeout or handle potential hang in download
                _shared_model = SentenceTransformer(model_name)
                logger.info("✅ Local embedding model loaded successfully.")
                break
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"❌ Failed to load SentenceTransformer model on attempt {attempt + 1}: {e}")
                logger.debug(f"Full traceback: {error_detail}")
                if attempt < max_retries:
                    logger.info("Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    logger.error("❌ All attempts to load embedding model failed.")
                    _shared_model = None
    return _shared_model


class EmbeddingService:
    """Service for generating and managing vector embeddings using local Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service. The model itself is a shared singleton.
        """
        self.model_name = model_name

    @property
    def model(self):
        """Lazy access to the shared model instance."""
        return get_shared_model(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using local model.
        
        Parameters
        ----------
        text : str
            Text to embed
            
        Returns
        -------
        List[float]
            Embedding vector
        """
        if not self.model:
            logger.error("Embedding model not loaded.")
            return []
            
        try:
            if not text or not text.strip():
                return []
            
            # encode returns a numpy array, convert to list
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating local embedding: {e}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch) using local model.
        
        Parameters
        ----------
        texts : List[str]
            Texts to embed
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
        """
        if not self.model:
            logger.error("Embedding model not loaded.")
            return []
            
        try:
            if not texts:
                return []
            
            # Filter empty texts
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return [[] for _ in texts]
            
            # Batch encode
            embeddings = self.model.encode(valid_texts)
            
            # Map back to original list size (with empty lists for filtered items)
            result = []
            valid_idx = 0
            for t in texts:
                if t and t.strip():
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append([])
            return result
        except Exception as e:
            logger.error(f"Error generating local batch embeddings: {e}")
            return [[] for _ in texts]

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Parameters
        ----------
        embedding1, embedding2 : List[float]
            Embedding vectors
            
        Returns
        -------
        float
            Similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None or len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        arr1 = np.array(embedding1, dtype=np.float32)
        arr2 = np.array(embedding2, dtype=np.float32)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def find_most_similar_faiss(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings using FAISS.
        
        Parameters
        ----------
        query_embedding : List[float]
            Query embedding vector
        candidate_embeddings : List[List[float]]
            List of candidate embeddings
        top_k : int
            Return top k most similar
            
        Returns
        -------
        List[tuple[int, float]]
            List of (index, similarity_score) tuples
        """
        if not candidate_embeddings or not query_embedding:
            return []
            
        # Convert to numpy arrays
        candidates_np = np.array(candidate_embeddings).astype('float32')
        query_np = np.array([query_embedding]).astype('float32')
        
        # FAISS uses L2 distance by default. For cosine similarity, 
        # we normalize vectors and use inner product.
        faiss.normalize_L2(candidates_np)
        faiss.normalize_L2(query_np)
        
        dimension = candidates_np.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product index
        index.add(candidates_np)
        
        # Search
        similarities, indices = index.search(query_np, min(top_k, len(candidate_embeddings)))
        
        return [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities[0]) if idx != -1]

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidates: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to query.
        Falls back to FAISS for larger candidate sets if available.
        """
        if HAS_FAISS and len(candidates) > 50:
            return self.find_most_similar_faiss(query_embedding, candidates, top_k)
            
        similarities = [
            (i, self.cosine_similarity(query_embedding, cand))
            for i, cand in enumerate(candidates)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def deduplicate_chunks(self, chunks: List[str], embeddings: List[List[float]], threshold: float = 0.9) -> List[str]:
        """
        Remove semantically redundant chunks.
        
        Parameters
        ----------
        chunks : List[str]
            Text chunks to deduplicate
        embeddings : List[List[float]]
            Embeddings for the chunks
        threshold : float
            Similarity threshold (0-1). Chunks with similarity > threshold are considered redundant.
            
        Returns
        -------
        List[str]
            Deduplicated chunks
        """
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return chunks
            
        unique_indices = []
        for i in range(len(embeddings)):
            is_redundant = False
            for j in unique_indices:
                sim = self.cosine_similarity(embeddings[i], embeddings[j])
                if sim > threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                unique_indices.append(i)
                
        return [chunks[i] for i in unique_indices]

    def rank_by_entities(self, chunks: List[str], entities: List[str]) -> List[tuple[int, float]]:
        """
        Boost ranking of chunks that contain tracked entities.
        
        Parameters
        ----------
        chunks : List[str]
            Text chunks to rank
        entities : List[str]
            Tracked entities from conversational context
            
        Returns
        -------
        List[tuple[int, float]]
            List of (index, boost_score) tuples
        """
        if not chunks or not entities:
            return [(i, 0.0) for i in range(len(chunks))]
            
        ranked = []
        for i, chunk in enumerate(chunks):
            boost = 0.0
            chunk_lower = chunk.lower()
            for entity in entities:
                if entity.lower() in chunk_lower:
                    boost += 0.2 # Boost for each entity match
            ranked.append((i, min(boost, 0.5))) # Cap boost at 0.5
            
        return ranked


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
