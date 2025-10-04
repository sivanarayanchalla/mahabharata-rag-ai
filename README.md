# 🕉️ Mahabharata RAG AI

A sophisticated AI-powered question-answering system for the Mahabharata epic, built with modern RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

- **Complete Mahabharata Knowledge**: All 18 parvas processed and searchable
- **AI-Powered Answers**: Powered by Llama 2 7B Chat model
- **Beautiful Web Interface**: Modern Streamlit UI with premium design
- **Source Citations**: Every answer includes references to original text
- **Real-time Performance**: Fast retrieval and generation

## 🛠️ Tech Stack

- **Backend**: Python, ChromaDB, Sentence Transformers
- **AI Model**: Llama 2 7B Chat (via Ollama)
- **Frontend**: Streamlit
- **Embeddings**: all-MiniLM-L6-v2
- **Vector Database**: ChromaDB

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/sivanarayanchalla/mahabharata-rag-ai.git
cd mahabharata-rag-ai

# Create virtual environment
python -m venv mahabharata_env
source mahabharata_env/bin/activate  # On Windows: mahabharata_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Llama 2 model
ollama pull llama2:7b-chat

# Start Ollama service (in separate terminal)
ollama serve

# Process all Mahabharata files
python run_complete_mahabharata.py

# Run the application
streamlit run app.py