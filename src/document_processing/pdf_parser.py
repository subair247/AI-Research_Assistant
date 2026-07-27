import fitz  # PyMuPDF
from typing import List, Dict, Any

class PDFParser:
    @staticmethod
    def extract_pages(pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        pages_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "doc_id": doc_id,
                    "page_number": page_num + 1,
                    "text": text
                })
        doc.close()
        return pages_data