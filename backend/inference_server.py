# backend/inference_server.py

from flask import Flask, request, jsonify
from src.vector_store import VectorStore
from src.rag_pipeline import ask_chatbot

# ---------------------------------------------------
# Load vector DB (FAISS or embeddings)
# ---------------------------------------------------
print("[SERVER] Loading VectorStore index from data/index ...")

vector_db = VectorStore()
vector_db.load("data/index")

print("[SERVER] Vector index loaded successfully.")


# ---------------------------------------------------
# Initialize Flask app
# ---------------------------------------------------
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided."}), 400

        # Run RAG pipeline
        answer = ask_chatbot(question, vector_db)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------
# Run server
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
