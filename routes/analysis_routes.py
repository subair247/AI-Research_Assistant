from flask import Blueprint, request, jsonify
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.database.base import SessionLocal
from src.database.models import DocumentModel

analysis_bp = Blueprint("analysis", __name__)

@analysis_bp.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    doc_id = data.get("doc_id")
    
    db = SessionLocal()
    doc = db.query(DocumentModel).filter_by(doc_id=doc_id).first()
    db.close()

    if not doc:
        return jsonify({"error": "Document not found"}), 404

    # Read extracted text from file path or chunks (simplified via file read)
    import fitz
    fitz_doc = fitz.open(doc.file_path)
    full_text = "".join([page.get_text() for page in fitz_doc])
    fitz_doc.close()

    summarizer = DocumentSummarizer()
    summary = summarizer.generate_summary(full_text)
    return jsonify(summary), 200

@analysis_bp.route("/compare", methods=["POST"])
def compare():
    data = request.get_json()
    doc_ids = data.get("doc_ids", [])
    
    db = SessionLocal()
    docs = db.query(DocumentModel).filter(DocumentModel.doc_id.in_(doc_ids)).all()
    db.close()

    doc_texts = {}
    import fitz
    for d in docs:
        f = fitz.open(d.file_path)
        doc_texts[d.file_name] = "".join([page.get_text() for page in f])
        f.close()

    comparator = DocumentComparator()
    comparison = comparator.compare_docs(doc_texts)
    return jsonify(comparison), 200