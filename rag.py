import re
import pickle
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from logger import logger


# Suppress lxml warning
import warnings
warnings.filterwarnings('ignore', message='.*lxml.*does not provide the extra.*')

class SimpleRAGSystem:
    """Simple RAG system using TF-IDF for document retrieval (now metadata-aware)."""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[str] = []
        self.chunk_meta: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self.is_fitted = False

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks (word-based, backward-compatible)."""
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        return chunks

    def _paragraph_chunks(self, text: str, target_words: int = 512, min_words: int = 120) -> List[str]:
        """Paragraph-aware chunker that targets ~512 'tokens' (approximated by words)"""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        chunks = []
        buf_words: List[str] = []
        for p in paragraphs:
            pw = p.split()
            if len(pw) >= target_words * 1.2:
                if buf_words:
                    buf = ' '.join(buf_words).strip()
                    if len(buf.split()) >= min_words:
                        chunks.append(buf)
                    buf_words = []
                for i in range(0, len(pw), target_words):
                    seg = pw[i:i + target_words]
                    if len(seg) >= min_words:
                        chunks.append(' '.join(seg).strip())
                continue

            if len(buf_words) + len(pw) <= target_words:
                buf_words.extend(pw)
            else:
                buf = ' '.join(buf_words).strip()
                if len(buf.split()) >= min_words:
                    chunks.append(buf)
                buf_words = pw[:]

        if buf_words:
            buf = ' '.join(buf_words).strip()
            if len(buf.split()) >= min_words:
                chunks.append(buf)
        return chunks

    def build_index(self, knowledge_base: str):
        """Build the RAG index from a single KB string (legacy behavior, no metadata)."""
        logger.info("Building RAG index (legacy KB string)...")
        self.chunks = self.chunk_text(knowledge_base)
        self.chunk_meta = [{"source_title": None, "source_url": None} for _ in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        logger.info("RAG index built successfully (legacy).")

    def build_index_from_blocks(self, blocks: List[Dict[str, str]]):
        """
        Build index from structured blocks:
        each block = {'text': str, 'source_title': str, 'source_url': str}
        """
        logger.info("Building RAG index from structured blocks (paragraph-aware)...")
        all_chunks, all_meta = [], []
        for b in blocks:
            text = (b.get("text") or "").strip()
            if not text:
                continue
            p_chunks = self._paragraph_chunks(text, target_words=max(self.chunk_size, 512))
            for ch in p_chunks:
                all_chunks.append(ch)
                all_meta.append({
                    "source_title": b.get("source_title"),
                    "source_url": b.get("source_url"),
                })

        if not all_chunks:
            logger.warning("No chunks produced from blocks; RAG index will be empty.")
            self.chunks, self.chunk_meta = [], []
            self.chunk_vectors, self.is_fitted = None, False
            return

        self.chunks = all_chunks
        self.chunk_meta = all_meta
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        logger.info(f"RAG index built with {len(self.chunks)} chunks (with metadata).")

    def retrieve_relevant_chunks(self, query: str, top_k=3) -> List[str]:
        """Backward-compatible: return only chunk texts (no metadata)."""
        if not self.is_fitted:
            return []
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx if sims[i] > 0.1]

    def retrieve_relevant_chunks_with_meta(self, query: str, top_k=3) -> List[Dict[str, Any]]:
        """Return chunks WITH metadata for citation-friendly prompts"""
        if not self.is_fitted:
            return []
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_idx = np.argsort(sims)[-top_k:][::-1]
        out = []
        for i in top_idx:
            if sims[i] > 0.1:
                out.append({
                    "text": self.chunks[i],
                    "source_title": self.chunk_meta[i].get("source_title"),
                    "source_url": self.chunk_meta[i].get("source_url"),
                    "similarity": float(sims[i]),
                })
        return out

    def save_index(self, filepath='rag_index.pkl'):
        """Save the RAG index + metadata to disk"""
        index_data = {
            'chunks': self.chunks,
            'chunk_meta': self.chunk_meta,
            'vectorizer': self.vectorizer,
            'chunk_vectors': self.chunk_vectors,
            'is_fitted': self.is_fitted,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        logger.info(f"RAG index saved to {filepath}")

    def load_index(self, filepath='rag_index.pkl'):
        """Load the RAG index from disk"""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            self.chunks = index_data.get('chunks', [])
            self.chunk_meta = index_data.get('chunk_meta', [{"source_title": None, "source_url": None} for _ in self.chunks])
            self.vectorizer = index_data['vectorizer']
            self.chunk_vectors = index_data['chunk_vectors']
            self.is_fitted = index_data['is_fitted']
            self.chunk_size = index_data.get('chunk_size', self.chunk_size)
            self.overlap = index_data.get('overlap', self.overlap)
            logger.info(f"RAG index loaded from {filepath}")
            return True
        except FileNotFoundError:
            logger.info(f"No existing index found at {filepath}")
            return False
