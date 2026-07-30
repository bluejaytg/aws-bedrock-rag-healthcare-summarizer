from typing import List, Dict, Any
import numpy as np

class InMemoryVectorStore:
    """Simple vector store mock representing OpenSearch Serverless / Pgvector execution."""

    def __init__(self):
        self.store: List[Dict[str, Any]] = []

    def add_vectors(self, vectors: List[List[float]], documents: List[str]):
        for vec, doc in zip(vectors, documents):
            self.store.append({"vector": np.array(vec), "document": doc})

    def similarity_search(self, query_vector: List[float], top_k: int = 3) -> List[str]:
        if not self.store:
            return []
        
        q_vec = np.array(query_vector)
        scored_docs = []
        for item in self.store:
            # Cosine similarity calculation
            sim = np.dot(q_vec, item["vector"]) / (np.linalg.norm(q_vec) * np.linalg.norm(item["vector"]))
            scored_docs.append((sim, item["document"]))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]