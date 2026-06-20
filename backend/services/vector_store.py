
import os
import pickle
import numpy as np
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_dir: str = "./vectorstore"):
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, "faiss.index")
        self.metadata_path = os.path.join(persist_dir, "metadata.pkl")

        # Lazy loading - model is NOT loaded during startup
        self.model = None

        self.dimension = 384
        self.index = None
        self.metadata: List[Dict] = []

        os.makedirs(persist_dir, exist_ok=True)
        self._load_or_create_index()

    def get_model(self):
        """Load model only when needed"""
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = self.get_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.astype(np.float32)

    def add_documents(self, chunks: List[Dict], filename: str) -> List[int]:
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._encode(texts)

        start_idx = len(self.metadata)
        self.index.add(embeddings)

        doc_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = start_idx + i
            self.metadata.append({
                **chunk,
                "doc_id": doc_id
            })
            doc_ids.append(doc_id)

        self._save_index()
        return doc_ids

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_embedding = self._encode([query])

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and 0 <= idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_document_count(self) -> int:
        return self.index.ntotal if self.index else 0

    def list_documents(self) -> List[Dict]:
        seen_files = {}

        for meta in self.metadata:
            fname = meta.get("filename", "unknown")

            if fname not in seen_files:
                seen_files[fname] = {
                    "filename": fname,
                    "chunks": 0,
                    "pages": set()
                }

            seen_files[fname]["chunks"] += 1

            if meta.get("page"):
                seen_files[fname]["pages"].add(meta["page"])

        result = []

        for fname, info in seen_files.items():
            result.append({
                "filename": fname,
                "chunks": info["chunks"],
                "pages": max(info["pages"]) if info["pages"] else None
            })

        return result

    def delete_document(self, filename: str) -> bool:
        indices_to_remove = [
            i for i, m in enumerate(self.metadata)
            if m.get("filename") == filename
        ]

        if not indices_to_remove:
            return False

        new_metadata = [
            m for m in self.metadata
            if m.get("filename") != filename
        ]

        keep_indices = [
            i for i in range(len(self.metadata))
            if i not in set(indices_to_remove)
        ]

        if keep_indices:
            all_texts = [m["text"] for m in self.metadata]
            all_embeddings = self._encode(all_texts)

            kept_embeddings = all_embeddings[keep_indices]

            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(kept_embeddings)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.metadata = new_metadata

        for i, meta in enumerate(self.metadata):
            meta["doc_id"] = i

        self._save_index()
        return True

    def clear_all(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save_index()
