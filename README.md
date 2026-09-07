# Intelligent PDF Q&A Chatbot using RAG

An intelligent PDF question-answering chatbot built using Python, LangChain, RAG, FAISS, Hugging Face embeddings, Ollama, and Streamlit.

## Features

- Upload a PDF file
- Extract and split PDF text
- Create embeddings using Hugging Face
- Store document vectors using FAISS
- Retrieve relevant PDF content
- Generate answers using a local Ollama model
- Display source pages
- Interactive and colorful Streamlit interface

## Technologies Used

- Python
- LangChain
- Streamlit
- Hugging Face Embeddings
- FAISS
- Ollama
- PyPDF

## RAG Workflow

PDF → Text Splitting → Embeddings → FAISS → Similarity Search → Ollama → Answer

## How to Run

### 1. Download the Project

Download or clone this repository to your computer.

### 2. Create and Activate Virtual Environment

Open the VS Code terminal and run:

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
ollama pull qwen2.5:0.5b
ollama run qwen2.5:0.5b
python -m streamlit run app.py