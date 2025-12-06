# src/vector_store.py

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


class VectorStore:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        self.model = SentenceTransformer(embed_model_name)

        self.embeddings = None
        self.metadatas = []
        self.chunk_files = []
        self.index = None

    # ----------------------------------------------------
    # BUILD INDEX (first time)
    # ----------------------------------------------------
    def build(self, chunk_json_paths: List[str]):
        if not chunk_json_paths:
            raise ValueError("No chunk files provided to build()")

        texts = []
        metas = []

        for p in chunk_json_paths:
            obj = json.loads(Path(p).read_text(encoding="cp1252"))
            texts.append(obj.get("text", ""))
            metas.append(obj.get("meta", {}))

        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # normalize for cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms

        self.embeddings = emb
        self.metadatas = metas
        self.chunk_files = list(chunk_json_paths)

        if faiss is not None:
            d = emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(emb)
        else:
            self.index = None

    # ----------------------------------------------------
    # SAVE INDEX
    # ----------------------------------------------------
    def save(self, save_dir: str = "data/index"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index or embeddings
        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / "faiss.index"))
        else:
            np.save(save_dir / "embeddings.npy", self.embeddings)

        # Save metadata
        (save_dir / "metadata.json").write_text(
            json.dumps(self.metadatas, ensure_ascii=False)
        )

        # Save chunk files mapping
        (save_dir / "chunk_files.json").write_text(
            json.dumps(self.chunk_files, ensure_ascii=False)
        )

        print(f"[VectorStore] Saved index & metadata to {save_dir}")

    # ----------------------------------------------------
    # LOAD INDEX
    # ----------------------------------------------------
    def load(self, load_dir: str = "data/index"):
        load_dir = Path(load_dir)

        # Load FAISS index if available
        faiss_index_path = load_dir / "faiss.index"
        emb_path = load_dir / "embeddings.npy"

        if faiss is not None and faiss_index_path.exists():
            self.index = faiss.read_index(str(faiss_index_path))
        else:
            self.index = None
            if emb_path.exists():
                self.embeddings = np.load(emb_path)
        
        # Load metadata
        meta_path = load_dir / "metadata.json"
        self.metadatas = json.loads(meta_path.read_text())

        # Load chunk files list
        chunk_files_path = load_dir / "chunk_files.json"
        self.chunk_files = json.loads(chunk_files_path.read_text())

        print(f"[VectorStore] Loaded index & metadata from {load_dir}")

    # ----------------------------------------------------
    # SEARCH
    # ----------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

        # FAISS search
        if self.index is not None:
            D, I = self.index.search(q_emb, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                results.append({
                    "score": float(score),
                    "meta": self.metadatas[idx],
                    "index": int(idx)
                })
            return results
        
        # Brute-force search
        sims = (self.embeddings @ q_emb.T).squeeze(1)
        best_idx = np.argsort(-sims)[:top_k]
        return [
            {
                "score": float(sims[i]),
                "meta": self.metadatas[i],
                "index": int(i)
            }
            for i in best_idx
        ]

    # ----------------------------------------------------
    # FETCH TEXT
    # ----------------------------------------------------
    def get_chunk_text(self, idx: int) -> str:
        p = self.chunk_files[idx]
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        return obj.get("text", "")
    
    def similarity_search(self, query: str, k: int = 5):
        """Wrapper to match LangChain-like similarity_search API."""
        results = self.search(query, top_k=k)
        out = []
    
        for r in results:
            idx = r["index"]
            text = self.get_chunk_text(idx)
            out.append(
                type("Obj", (object,), {
                    "page_content": text,
                    "metadata": r["meta"],
                    "score": r["score"]
                })()
            )
        return out




