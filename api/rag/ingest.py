import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import faiss
import numpy as np
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "user_manual.pdf"
INDEX_DIR = DATA_DIR / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.pkl"


def extract_text_from_pdf() -> List[Dict[str, Any]]:
    """Extract text from PDF page by page"""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    logger.info(f"Extracting text from {PDF_PATH}")
    pages_data = []

    try:
        doc = fitz.open(PDF_PATH)
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                pages_data.append(
                    {
                        "text": text,
                        "page": page_num,
                        "source": f"user_manual.pdf",
                    }
                )
        doc.close()
        logger.info(f"Extracted {len(pages_data)} pages")
        return pages_data
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def chunk_pages(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk pages using RecursiveCharacterTextSplitter"""
    logger.info("Chunking pages...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for page_data in pages_data:
        text = page_data["text"]
        page_num = page_data["page"]
        source = page_data["source"]

        split_texts = splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_num,
                    "source": source,
                    "chunk_id": f"{page_num}_{chunk_idx}",
                }
            )

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def get_embeddings():
    """Get embeddings based on provider"""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "openai":
        logger.info("Using OpenAI embeddings (text-embedding-3-small)")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_faiss_index(chunks: List[Dict[str, Any]]) -> tuple:
    """Create FAISS index from chunks"""
    logger.info("Creating FAISS index...")

    embeddings = get_embeddings()

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings_list = embeddings.embed_documents(texts)
    embeddings_array = np.array(embeddings_list).astype("float32")

    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    logger.info(f"FAISS index created with {index.ntotal} embeddings")
    return index, chunks


def save_index(index: faiss.IndexFlatL2, metadata: List[Dict[str, Any]]):
    """Save FAISS index and metadata"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(INDEX_FILE))
    logger.info(f"Index saved to {INDEX_FILE}")

    # Save metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to {METADATA_FILE}")


def ingest_pdf() -> int:
    """Main ingestion pipeline"""
    try:
        # Extract pages
        pages = extract_text_from_pdf()

        # Chunk pages
        chunks = chunk_pages(pages)

        # Create FAISS index
        index, chunk_metadata = create_faiss_index(chunks)

        # Save index and metadata
        save_index(index, chunk_metadata)

        logger.info(f"Ingestion complete: {len(chunks)} chunks indexed")
        return len(chunks)

    except Exception as e:
        logger.error(f"Ingest pipeline failed: {e}")
        raise


if __name__ == "__main__":
    ingest_pdf()
