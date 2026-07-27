from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        chunk_counter = 0
        for page in pages_data:
            text = page["text"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append({
                    "chunk_id": f"{page['doc_id']}_c{chunk_counter}",
                    "doc_id": page["doc_id"],
                    "page_number": page["page_number"],
                    "text": chunk_text
                })
                chunk_counter += 1
                start += (self.chunk_size - self.chunk_overlap)
        return chunks