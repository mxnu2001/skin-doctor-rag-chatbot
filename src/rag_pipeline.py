# src/rag_pipeline.py
import glob
import json
from pathlib import Path
from typing import List
from .vector_store import VectorStore
from .model_loader import load_model

from src.model_loader import generate_answer
from src.relevance_checker import is_skin_related


def ask_chatbot(query: str, vector_db):
    if not is_skin_related(query):
        return "Please ask only skin-related questions."

    results = vector_db.similarity_search(query, k=2)
    if not results:
        return "I don’t have this information right now. Please contact the doctor."

    context = "\n".join([f"Q: {r.metadata['question']} A: {r.page_content}" for r in results])
    prompt = (
        "Answer based ONLY on the context below. "
        "If unsure, say you don't have the info.\n\n"
        f"{context}\n\nPatient: {query}\nDoctor:"
    )

    answer = generate_answer(prompt)

    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip().startswith(("Patient:", "Doctor:"))
    ]

    return "\n".join(lines)
