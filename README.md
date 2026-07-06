# 🩺 Skin Doctor RAG Chatbot

An AI-powered dermatology assistant built using **Retrieval-Augmented Generation (RAG)** that answers skin-related questions using a curated medical knowledge base instead of relying solely on a Large Language Model.

The application combines **Mistral-7B**, **Sentence Transformers**, and **FAISS** to retrieve relevant medical information before generating context-aware responses.

---

## ✨ Features

- 💬 Conversational chatbot for skin-related queries
- 🔍 Retrieval-Augmented Generation (RAG) pipeline
- 🧠 Mistral-7B-Instruct for response generation
- 📚 Semantic search using Sentence Transformers
- ⚡ Fast vector similarity search with FAISS
- 🌐 Flask REST API backend
- 🎨 Streamlit-based user interface
- 🚫 Rejects non-dermatology questions
- 📄 CSV preprocessing and chunk generation
- 📦 Modular project architecture

---

# System Architecture

```
                 User
                  │
                  ▼
        Streamlit Web Interface
                  │
                  ▼
           Flask REST API
                  │
                  ▼
         Relevance Checker
                  │
        Skin Related?
          │          │
         No         Yes
          │          ▼
   Reject Query   FAISS Search
                      │
                      ▼
          Retrieve Relevant Context
                      │
                      ▼
            Prompt Construction
                      │
                      ▼
            Mistral-7B-Instruct
                      │
                      ▼
              Generated Response
                      │
                      ▼
                 User Interface
```

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Large Language Model (LLM)
- Mistral-7B-Instruct

### Embedding Model
- Sentence Transformers (`all-MiniLM-L6-v2`)

### Vector Database
- FAISS

### Backend Framework
- Flask

### Frontend Framework
- Streamlit

### API
- REST API

### Machine Learning Frameworks
- Hugging Face Transformers
- Sentence Transformers

### Model Hosting
- Hugging Face Hub

---

# Project Structure

```
skin-doctor-rag-chatbot/

│
├── backend/
│   └── inference_server.py
│
├── src/
│   ├── model_loader.py
│   ├── preprocess.py
│   ├── rag_pipeline.py
│   ├── relevance_checker.py
│   └── vector_store.py
│
├── streamlit_ui/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# How It Works

## 1. Data Preparation

Medical data stored in CSV format is cleaned and split into overlapping chunks.

```
CSV
    ↓
Text Cleaning
    ↓
Chunking
    ↓
JSON Files
```

---

## 2. Embedding Generation

Each chunk is converted into a semantic embedding using:

- sentence-transformers/all-MiniLM-L6-v2

These embeddings are indexed inside FAISS.

---

## 3. Query Processing

When a user asks a question:

- Check whether the question is related to dermatology.
- Reject unrelated questions.
- Convert the query into an embedding.
- Retrieve the most relevant chunks using FAISS.
- Construct a prompt containing the retrieved context.
- Generate the final response using Mistral-7B.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/mxnu2001/skin-doctor-rag-chatbot.git

cd skin-doctor-rag-chatbot
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```
HF_TOKEN=your_huggingface_token
```

---

# Running the Backend

```bash
python backend/inference_server.py
```

Backend starts on

```
http://localhost:5000
```

---

# Running the Streamlit UI

```bash
streamlit run streamlit_ui/app.py
```

---

# API Endpoint

### POST

```
/ask
```

Request

```json
{
    "question":"How can I treat acne?"
}
```

Response

```json
{
    "answer":"..."
}
```

---

# Retrieval-Augmented Generation Pipeline

```
User Query
      │
      ▼
Relevance Check
      │
      ▼
Sentence Transformer
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top Relevant Chunks
      │
      ▼
Prompt Construction
      │
      ▼
Mistral-7B
      │
      ▼
Final Response
```

---

# Current Capabilities

- Dermatology question answering
- Semantic document retrieval
- Context-aware response generation
- REST API inference
- Interactive chat interface

---

# Future Improvements

- Doctor appointment scheduling
- Image-based skin disease detection
- Multi-language support
- Conversation memory
- Medical citation support
- Deployment on AWS/Azure
- PostgreSQL knowledge storage
- User authentication

---

# Dependencies

- transformers
- sentence-transformers
- faiss-cpu
- flask
- streamlit
- requests
- accelerate
- bitsandbytes
- python-dotenv
- pyngrok

---

# Author

**Manu Pradeep Kumar**

B.Tech Computer Science (AI & Robotics)

Interested in

- Artificial Intelligence
- NLP
- Large Language Models
- Retrieval-Augmented Generation
- Machine Learning
- Data Science

---

# License

This project is intended for educational and research purposes.

It is **not** a substitute for professional medical advice.
