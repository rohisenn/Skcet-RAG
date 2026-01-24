# 🎓 SKCET Knowledge Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) based AI assistant built for  
**Sri Krishna College of Engineering and Technology (SKCET)**.

This system enables users to ask natural language questions about SKCET and receive accurate, context-aware answers powered by document embeddings and a Large Language Model (Groq).

🔗 **Live App:** https://skcet1.streamlit.app/

---

## 📌 Problem Statement

College-related information (history, achievements, facilities, administration, etc.) is scattered across multiple web pages and documents.  
Students and visitors often struggle to find accurate answers quickly.

**Goal:**  
Build an intelligent assistant that can:
- Understand natural language queries
- Retrieve relevant information from SKCET documents
- Generate clear, concise answers
- Ensure data security and reliability

---

## 🧠 Solution Overview

This project uses a **Retrieval-Augmented Generation (RAG)** architecture:

1. Documents are converted into vector embeddings (offline)
2. Embeddings are stored in a vector database (Chroma)
3. User queries retrieve relevant chunks
4. A Large Language Model generates the final answer
