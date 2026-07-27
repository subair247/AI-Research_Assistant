from src.database.models import DocumentModel
from sqlalchemy.orm import Session

class AnalyticsEngine:
    @staticmethod
    def get_metrics(session: Session) -> dict:
        total_docs = session.query(DocumentModel).count()
        total_chunks = session.query(DocumentModel).with_entities(DocumentModel.total_chunks).all()
        sum_chunks = sum([c[0] for c in total_chunks if c[0]])
        
        categories = session.query(DocumentModel.category).all()
        cat_counts = {}
        for cat in categories:
            c = cat[0]
            cat_counts[c] = cat_counts.get(c, 0) + 1

        return {
            "total_documents": total_docs,
            "total_processed_chunks": sum_chunks,
            "category_distribution": cat_counts
        }