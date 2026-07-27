from flask import Blueprint, request, jsonify, current_app
from src.rag.qa_chain import RAGQuestionAnswering
from src.database.base import SessionLocal
from src.database.models import ChatSessionModel

search_bp = Blueprint("search", __name__)

@search_bp.route("/qa", methods=["POST"])
def question_answering():
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id", "default_session")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    db = SessionLocal()
    session_rec = db.query(ChatSessionModel).filter_by(session_id=session_id).first()
    history = session_rec.history if session_rec else ""

    rag = RAGQuestionAnswering(current_app.config["VECTOR_DB_DIR"])
    result = rag.answer(query, history)

    # Update conversation history
    new_history = f"{history}\nUser: {query}\nAssistant: {result['answer']}"
    if session_rec:
        session_rec.history = new_history
    else:
        session_rec = ChatSessionModel(session_id=session_id, history=new_history)
        db.add(session_rec)
    db.commit()
    db.close()

    return jsonify(result), 200