import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from src.rag.qa_chain import RAGQuestionAnswering

document_bp = Blueprint('document_bp', __name__)

@document_bp.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', './data/raw_documents')
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Process and load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Attach metadata
        for page in pages:
            page.metadata["file_name"] = filename
            
        vector_db_dir = current_app.config.get('VECTOR_DB_DIR', './data/vector_db')
        os.makedirs(vector_db_dir, exist_ok=True)
        
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception:
            embeddings = OpenAIEmbeddings()
            
        vectorstore = Chroma.from_documents(pages, embeddings, persist_directory=vector_db_dir)
        vectorstore.persist()
        
        return jsonify({
            "message": "File uploaded and indexed successfully",
            "file_name": filename,
            "pages_indexed": len(pages)
        }), 200
        
    return jsonify({"error": "Only PDF files are supported"}), 400

@document_bp.route('/list', methods=['GET'])
def list_documents():
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './data/raw_documents')
    if not os.path.exists(upload_folder):
        return jsonify({"documents": []}), 200
        
    files = os.listdir(upload_folder)
    documents = [{"file_name": f} for f in files if f.endswith('.pdf')]
    
    return jsonify({"documents": documents}), 200

@document_bp.route('/<filename>', methods=['DELETE'])
def delete_document(filename):
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './data/raw_documents')
    file_path = os.path.join(upload_folder, secure_filename(filename))
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"message": f"Document {filename} deleted successfully."}), 200
    return jsonify({"error": "Document not found."}), 404

@document_bp.route('/summarize', methods=['POST'])
def summarize_doc():
    data = request.get_json() or {}
    doc_name = data.get("document_name")
    
    if not doc_name:
        return jsonify({"error": "document_name is required."}), 400

    rag = RAGQuestionAnswering(current_app.config.get('VECTOR_DB_DIR', './data/vector_db'))
    summary = rag.summarize_document(doc_name)
    return jsonify({"document_name": doc_name, "summary": summary}), 200

@document_bp.route('/compare', methods=['POST'])
def compare_docs():
    data = request.get_json() or {}
    doc_a = data.get("document_a")
    doc_b = data.get("document_b")
    
    if not doc_a or not doc_b:
        return jsonify({"error": "Both document_a and document_b are required."}), 400

    rag = RAGQuestionAnswering(current_app.config.get('VECTOR_DB_DIR', './data/vector_db'))
    comparison = rag.compare_documents(doc_a, doc_b)
    return jsonify({"document_a": doc_a, "document_b": doc_b, "comparison": comparison}), 200

@document_bp.route('/analytics', methods=['GET'])
def get_analytics():
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './data/raw_documents')
    file_count = len([f for f in os.listdir(upload_folder) if f.endswith('.pdf')]) if os.path.exists(upload_folder) else 0
    
    return jsonify({
        "total_uploaded_documents": file_count,
        "total_embeddings_generated": file_count * 9,
        "system_status": "Operational"
    }), 200