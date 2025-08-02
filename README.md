PDF Question-Answer Generator
An intelligent web application that automatically generates questions and answers from PDF documents using GROQ AI models and RAG (Retrieval-Augmented Generation) technology.

🚀 Features

PDF Upload: Upload any PDF document through a web interface
Automatic Question Generation: AI-powered question creation from document content
Intelligent Answer Generation: RAG-based answer generation using vector embeddings
CSV Export: Download generated Q&A pairs as CSV file
Fast Processing: Powered by GROQ's high-speed inference
Free to Use: Uses HuggingFace embeddings (no OpenAI API required)

🛠️ Tech Stack

Backend: FastAPI (Python)
AI Models: GROQ (Llama3-8B-8192)
Embeddings: HuggingFace Sentence Transformers
Vector Store: FAISS
Document Processing: LangChain
Frontend: HTML/CSS/JavaScript with Jinja2 templates

📋 Prerequisites

Python 3.8+
GROQ API Key (free at console.groq.com)

### How to run?

1. Create an environment

```bash
conda create -n interview python=3.11 -y


conda activate interview

```

2. install requirements

```bash
pip install -r requirements.txt
```

🙏 Acknowledgments
-GROQ for providing fast AI inference
-LangChain for RAG framework
-HuggingFace for free embeddings
-FastAPI for the web framework

🤝 Contributing
-Fork the repository
-Create a feature branch (git checkout -b feature/new-feature)
-Commit your changes (git commit -am 'Add new feature')
-Push to the branch (git push origin feature/new-feature)
-Create a Pull Request