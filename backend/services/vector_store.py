import os
import pickle
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self, persist_dir: str = "./vectorstore"):
        self.persist_dir = persist_dir
        self.metadata_path = os.path.join(persist_dir, "metadata.pkl")

        os.makedirs(persist_dir, exist_ok=True)

        self.metadata: List[Dict] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = None

        self._load()

    def _load(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            if self.metadata:
                texts = [m["text"] for m in self.metadata]
                self.document_vectors = self.vectorizer.fit_transform(texts)

    def _save(self):
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _rebuild_index(self):
        if self.metadata:
            texts = [m["text"] for m in self.metadata]
            self.document_vectors = self.vectorizer.fit_transform(texts)
        else:
            self.document_vectors = None

    def add_documents(self, chunks: List[Dict], filename: str) -> List[int]:
        start_idx = len(self.metadata)

        doc_ids = []

        for i, chunk in enumerate(chunks):
            doc_id = start_idx + i

            self.metadata.append({
                **chunk,
                "doc_id": doc_id
            })

            doc_ids.append(doc_id)

        self._rebuild_index()
        self._save()

        return doc_ids

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.metadata or self.document_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []

        for idx in ranked_indices:
            score = float(similarities[idx])

            result = self.metadata[idx].copy()
            result["score"] = score

            results.append(result)

        return results

    def get_document_count(self) -> int:
        return len(self.metadata)

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
        new_metadata = [
            m for m in self.metadata
            if m.get("filename") != filename
        ]

        if len(new_metadata) == len(self.metadata):
            return False

        self.metadata = new_metadata

        for i, meta in enumerate(self.metadata):
            meta["doc_id"] = i

        self._rebuild_index()
        self._save()

        return True

    def clear_all(self):
        self.metadata = []
        self.document_vectors = None
        self._save()
