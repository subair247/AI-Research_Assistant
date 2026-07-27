from flask import Blueprint, jsonify
from src.database.base import SessionLocal
from src.analytics.metrics import AnalyticsEngine

analytics_bp = Blueprint("analytics", __name__)

@analytics_bp.route("/stats", methods=["GET"])
def get_stats():
    db = SessionLocal()
    metrics = AnalyticsEngine.get_metrics(db)
    db.close()
    return jsonify(metrics), 200