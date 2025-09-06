# 🌍 Multilingual NLP Toolkit + 🤖 Domain QA Chatbot  
> **NER · Summarization · Translation · Conversational RAG with ChromaDB**

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-HuggingFace-orange?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/RAG-ChromaDB-purple" />
  <img src="https://img.shields.io/badge/LLM-Groq-green" />
  <img src="https://img.shields.io/badge/Visualization-Matplotlib-blue" />
</p>

---

## ✨ Overview

This repository contains **two Streamlit-based applications** for applied NLP and Question Answering:

1. **🌍 Multilingual NLP Toolkit** (`Sam_544_ESE1.py`)  
   - 🧩 **Named Entity Recognition (NER)** — multilingual (Indic + global languages)  
   - 📝 **Summarization** — based on `mT5` trained on XL-Sum  
   - 🔁 **English → French Translation** — MarianMT models  
   - 📊 **Evaluation** — ROUGE, SacreBLEU, Precision/Recall/F1  

2. **🤖 Domain QA Chatbot** (`ESE2.py`)  
   - 📂 Load your **domain documents** into **ChromaDB**  
   - 🔎 **Conversational retrieval** with memory (context-aware Q&A)  
   - ⚡ Powered by **Groq LLM (`llama-3.1-8b-instant`)**  
   - 📊 Built-in analytics (conversation turns, response lengths, plots)

💡 With these apps, you can:  
- Run **multilingual NER, summarization & translation**  
- Evaluate NLP outputs with **standard metrics**  
- Build your own **chatbot over domain-specific documents** 🚀

---

## 📂 Project Structure
```bash
multilingual-nlp-qa/
│── Sam_544_ESE1.py     # Multilingual NLP Toolkit (NER, Summarization, Translation, Evaluation)
│── Sam_544_ESE2.py             # Domain-Specific QA Chatbot (RAG with ChromaDB + Groq LLM)
│── requirements.txt    # Python dependencies
│── README.md           # Documentation
│── chroma_store/       # Persistent ChromaDB vector store (auto-created)
