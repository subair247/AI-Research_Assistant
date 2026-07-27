import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, db_path: str):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if not os.path.isabs(db_path):
            self.db_path = os.path.join(base_dir, db_path)
        else:
            self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def add_chunks(self, chunks: list, file_name: str):
        texts = [c["text"] for c in chunks]
        metadatas = [{
            "doc_id": c["doc_id"],
            "page_number": c["page_number"],
            "file_name": file_name
        } for c in chunks]

        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
            db.add_texts(texts=texts, metadatas=metadatas)
        else:
            db = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)
        
        db.save_local(self.db_path)

    def search(self, query: str, k: int = 4):
        index_file = os.path.join(self.db_path, "index.faiss")
        if not os.path.exists(index_file):
            return []
            
        db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        return db.similarity_search(query, k=k)