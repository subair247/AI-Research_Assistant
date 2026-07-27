# AI Research Assistant

## Project Overview
The AI Research Assistant is a modular, full-stack application designed to help researchers upload, classify, search, summarize, compare, and query academic or technical documents using Retrieval-Augmented Generation (RAG). It combines deep learning text classification with LLM-powered context analysis to streamline document management and literature review workflows.

---

## Architecture Diagram
```
+-----------------------------------------------------------------+
|                        Client / Frontend                        |
+-----------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|                        Flask Application                        |
|              (Document Routes & Search Routes API)              |
+-----------------------------------------------------------------+
        |                        |                        |
        v                        v                        v
+---------------+     +--------------------+     +------------------+----+
|   TensorFlow  |     |   FAISS Vector     |     |   SQLite & SQLAlchemy |
|   Classifier  |     |   Store Manager    |     |   Metadata Store |    |
+---------------+     +--------------------+     +------------------+----+
                                 |
                                 v
                     +-----------------------+
                     | Google Gemini / LLM   |
                     | (RAG, Summary, etc.)  |
                     +-----------------------+

```
## Technology Stack
Backend Framework: Flask, Flask-CORS

Database & ORM: SQLite, SQLAlchemy

Vector Database: FAISS (CPU)

Embeddings & LLM: LangChain, Sentence-Transformers, Google Gemini API (langchain-google-genai)

Machine Learning: TensorFlow / Keras

Document Processing: PyMuPDF (fitz)

Environment Configuration: Python-Dotenv, Pydantic

```
## Setup Instructions
1.Clone the Repository:
git clone <repository-url>
cd ai-research-assistant

2.Create and Activate a Virtual Environment:
python -m venv venv
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

3.Install Dependencies:
pip install -r requirements.txt
pip install langchain-google-genai

4.Configure Environment Variables:
Create a .env file in the root directory based on .env.example and add your API keys.

5.Run the Application:
python main.py
```
## Environment Variables
Create a .env file in the root folder containing the following:

GEMINI_API_KEY=your_gemini_api_key_here.
UPLOAD_FOLDER=./data/raw_documents.
VECTOR_DB_DIR=./data/vector_db.
MODEL_PATH=./models/tf_classifier.h5.
DATABASE_URL=sqlite:///./data/metadata.db.

---
``
## API Documentation
Document Routes (/api/documents)
POST /upload: Uploads a PDF document, extracts text via PyMuPDF, performs automatic ML text classification, chunks text, and stores vectors in FAISS.

GET /list: Retrieves metadata for all successfully uploaded and processed documents.

DELETE /delete/<id>: Deletes a document record and cleans up associated assets.
```
Search & RAG Routes (/api/search)
POST /query: Accepts a natural language query, retrieves relevant text snippets via FAISS vector search, and generates an answer using Google Gemini with precise citations.

POST /summarize: Generates an AI-driven summary for a specified document.

POST /compare: Compares content or key findings across multiple uploaded documents.
---
```
## Assumptions
Uploaded files are predominantly text-searchable PDF research papers.

The local file system has sufficient write permissions to store SQLite databases, FAISS vector indexes, and trained TensorFlow models locally under the data/ and models/ directories.

Google AI Studio API access is active and provisioned with standard rate limits for development and evaluation.

## Design Decisions

Modular Separation: Divided core logic into discrete modules (rag, ml, vector_store, analytics, routes) to ensure high testability, clean code boundaries, and simplified maintenance.

FAISS Vector Store: Selected for high-performance similarity search locally without requiring external cloud database dependencies.

Google Gemini Integration: Utilized Google's Gemini models via LangChain to leverage massive context windows and free-tier accessibility for robust research document handling.

TensorFlow Text Classification: Implemented local text classification to categorize documents autonomously upon upload.

## Limitations

OCR is not natively integrated, meaning scanned image-based PDFs without embedded text layers may yield empty extraction text.

Local vector storage using FAISS limits concurrent multi-process writes without locking constraints.

Large files can increase processing duration during initial text embedding and vector indexing phases.

## Future Improvements
Implement background worker queues (such as Celery with Redis) for asynchronous large PDF processing.

Add native OCR support using Tesseract for scanned document ingestion.

Expand the frontend user interface to provide a fully interactive web dashboard for research management.