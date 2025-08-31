"""Sentence transformers embedding integration for AAS LanceDB MCP."""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

from .models import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages sentence transformer models and embedding generation."""

    def __init__(self):
        self._models: Dict[str, SentenceTransformer] = {}
        self._configs: Dict[str, EmbeddingConfig] = {}

    @lru_cache(maxsize=5)
    def _load_model(self, model_name: str, device: str = "cpu") -> SentenceTransformer:
        """Load and cache a sentence transformer model."""
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            model = SentenceTransformer(model_name, device=device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model(self, config: EmbeddingConfig) -> SentenceTransformer:
        """Get or load a sentence transformer model."""
        cache_key = f"{config.model_name}_{config.device}"
        
        if cache_key not in self._models:
            self._models[cache_key] = self._load_model(config.model_name, config.device)
            self._configs[cache_key] = config
            
        return self._models[cache_key]

    def embed_text(
        self, 
        text: str, 
        config: EmbeddingConfig,
        normalize: Optional[bool] = None
    ) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text], config, normalize)[0]

    def embed_texts(
        self, 
        texts: List[str], 
        config: EmbeddingConfig,
        normalize: Optional[bool] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
            
        model = self.get_model(config)
        use_normalize = normalize if normalize is not None else config.normalize_embeddings
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            embeddings = model.encode(
                texts, 
                normalize_embeddings=use_normalize,
                show_progress_bar=len(texts) > 10
            )
            
            # Convert numpy arrays to lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            elif isinstance(embeddings, list):
                embeddings = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
                
            logger.debug(f"Generated embeddings with dimension: {len(embeddings[0])}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        try:
            # Create temporary config to load model
            temp_config = EmbeddingConfig(model_name=model_name)
            model = self.get_model(temp_config)
            
            return {
                "model_name": model_name,
                "max_seq_length": getattr(model, "max_seq_length", None),
                "dimension": model.get_sentence_embedding_dimension(),
                "device": str(model.device),
                "tokenizer": type(model.tokenizer).__name__ if hasattr(model, 'tokenizer') else None
            }
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear model cache."""
        logger.info("Clearing embedding model cache")
        self._models.clear()
        self._configs.clear()
        self._load_model.cache_clear()


# Global embedding manager instance
embedding_manager = EmbeddingManager()