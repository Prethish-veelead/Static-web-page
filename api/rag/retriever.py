import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import pickle

import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_FILE = DATA_DIR / "faiss_index" / "index.faiss"
METADATA_FILE = DATA_DIR / "faiss_index" / "metadata.pkl"


class RAGRetriever:
    """Singleton retriever for FAISS index"""

    def __init__(self):
        self.index = None
        self.metadata = []
        self.embeddings = None
        self._load_index()

    def _get_embeddings(self):
        """Get embeddings based on provider"""
        if self.embeddings is not None:
            return self.embeddings

        provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

        if provider == "openai":
            logger.info("Using OpenAI embeddings (text-embedding-3-small)")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        return self.embeddings

    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if INDEX_FILE.exists() and METADATA_FILE.exists():
                self.index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Index loaded with {self.index.ntotal} embeddings")
            else:
                logger.warning(f"Index files not found at {INDEX_FILE}")
        except Exception as e:
            logger.error(f"Error loading index: {e}")

    def is_index_loaded(self) -> bool:
        """Check if index is loaded"""
        return self.index is not None and self.index.ntotal > 0

    def get_chunk_count(self) -> int:
        """Get total number of indexed chunks"""
        return self.index.ntotal if self.index else 0

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for query"""
        if not self.is_index_loaded():
            raise RuntimeError("Run POST /api/ingest first to create the index")

        try:
            # Get embeddings
            embeddings = self._get_embeddings()

            # Generate query embedding
            query_embedding = embeddings.embed_query(query)
            query_array = np.array([query_embedding]).astype("float32")

            # Search FAISS index
            distances, indices = self.index.search(query_array, top_k)

            # Retrieve metadata
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    chunk_data = self.metadata[idx].copy()
                    results.append(chunk_data)

            logger.info(f"Retrieved {len(results)} chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
