# 📚 Knowledge Graph Chatbot

[![Hugging Face](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/spaces/samuelolubukun/KnowledgeGraphChatbot)

An interactive chatbot that lets you **ingest documents & URLs**, build a **knowledge graph in Neo4j**, and chat with your knowledge using **LLMs (Cohere / Gemini)**. It combines **structured graph queries** and **vector-based semantic search** to deliver accurate, source-grounded answers.

---

## 🚀 Features

* **Data Ingestion**: Upload PDFs, DOCX, TXT, CSV, XLSX or scrape web pages.
* **Knowledge Graph Extraction**: Uses LLMs to identify entities, relationships, and concepts.
* **Vector Search**: Embedding-based retrieval for contextual answers.
* **Hybrid QA**: Combines Cypher graph queries + vector semantic search.
* **Source Attribution**: Answers reference original documents/URLs.
* **Strict Mode**: No hallucinations — answers only from your ingested data.
* **Interactive UI**: Built with [Gradio](https://www.gradio.app/) for an easy-to-use interface.

---

## 🛠️ Tech Stack

* [LangChain](https://www.langchain.com/) for orchestration
* [Neo4j](https://neo4j.com/) for graph storage + vector index
* [Cohere](https://cohere.ai/) or [Google Gemini](https://ai.google.dev/) for LLM & embeddings
* [Gradio](https://www.gradio.app/) for UI
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) + Requests for web scraping

---

## ⚡ How It Works

1. **Connect to Neo4j** (AuraDB or self-hosted)
2. **Choose LLM Provider** (Cohere or Gemini) + enter API key
3. **Upload files or enter URLs**
4. **Process & Build Knowledge Graph** → creates entities, relationships, and vector index
5. **Ask Questions** → chatbot retrieves structured + unstructured knowledge to answer

---

## 📦 Installation (for local / Kaggle use)

```bash
pip install gradio langchain neo4j python-dotenv requests beautifulsoup4
pip install langchain-community langchain-experimental
pip install langchain-google-genai cohere
pip install json-repair
pip install unstructured[all-docs]
```

---

## ▶️ Run Locally

```bash
python app.py
```

The Gradio interface will launch at `http://0.0.0.0:7860` (or shareable link if `share=True`).

---

## 🌐 Hugging Face Space

👉 Try it directly on [Hugging Face Spaces](https://huggingface.co/spaces/samuelolubukun/KnowledgeGraphChatbot)

---

## 🔑 Requirements

* **Neo4j database** (Neo4j Aura recommended)
* **API Key** for [Cohere](https://dashboard.cohere.com/) or [Google Gemini](https://ai.google.dev/)
* Documents or URLs to process

---

## 📖 Example Queries

* *Entity-based*: “What organizations are mentioned in the documents?”
* *Relationship*: “How are \[Entity A] and \[Entity B] related?”
* *Content*: “Summarize the main concepts about cybersecurity in these files.”

---

## 👤 Author

Developed by **[samuelolubukun](https://github.com/samolubukun)**

---


