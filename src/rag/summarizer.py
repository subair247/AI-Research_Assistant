import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.vector_store.manager import VectorStoreManager

class DocumentSummarizer:
    def __init__(self, vector_db_dir: str):
        self.v_manager = VectorStoreManager(vector_db_dir)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0
        )

    def summarize_document(self, file_name: str):
        docs = self.v_manager.search(file_name, k=10)
        filtered_docs = [d for d in docs if d.metadata.get("file_name") == file_name]
        
        if not filtered_docs:
            return {"error": "Document not found or has no indexed content."}

        context_str = "\n".join([d.page_content for d in filtered_docs])

        template = """
        Provide a comprehensive and structured summary of the following research document content:
        
        Context:
        {context}

        Summary:
        """
        prompt = PromptTemplate(template=template, input_variables=["context"])
        formatted = prompt.format(context=context_str)
        response = self.llm.invoke(formatted)

        return {
            "file_name": file_name,
            "summary": response.content
        }