from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from src.database.base import Base

class DocumentModel(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String, default="PENDING")
    category = Column(String, default="Unclassified")
    file_path = Column(String, nullable=False)

class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True, index=True)
    history = Column(Text, default="")
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)