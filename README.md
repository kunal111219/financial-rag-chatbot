# 💹 Financial RAG Chatbot

An AI-powered chatbot that lets you chat with financial documents
(annual reports, 10-K filings, earnings calls) using RAG architecture.

## 🛠 Tech Stack

- **LLM:** Groq llama-3.1-8b-instant (fastest inference)
- **Framework:** LangChain
- **Embeddings:** HuggingFace sentence-transformers
- **Vector DB:** FAISS
- **Frontend:** Streamlit
- **Deployment:** Streamlit Cloud

## 🚀 Live Demo

[Click here to try it](https://financial-rag-chatbot-hvtehvgvo4a7toyypuwa2y.streamlit.app/)

## 📊 Features

- Upload any financial PDF
- Ask natural language questions
- Get precise answers with source citations
- Sub-2 second response time via Groq

## 🏃 Run Locally

pip install -r requirements.txt
streamlit run app.py
