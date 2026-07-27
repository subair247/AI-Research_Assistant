import os
import pytest
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker

def test_chunker_logic():
    sample_pages = [
        {"doc_id": "test_doc_1", "page_number": 1, "text": "A" * 1500}
    ]
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.create_chunks(sample_pages)
    
    assert len(chunks) > 1
    assert chunks[0]["doc_id"] == "test_doc_1"
    assert chunks[0]["page_number"] == 1
    assert len(chunks[0]["text"]) == 1000