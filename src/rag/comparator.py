import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.vector_store.manager import VectorStoreManager

class DocumentComparator:
    def __init__(self, vector_db_dir: str):
        self.v_manager = VectorStoreManager(vector_db_dir)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0
        )

    def compare_documents(self, file_name_1: str, file_name_2: str):
        docs_1 = [d.page_content for d in self.v_manager.search(file_name_1, k=5) if d.metadata.get("file_name") == file_name_1]
        docs_2 = [d.page_content for d in self.v_manager.search(file_name_2, k=5) if d.metadata.get("file_name") == file_name_2]

        if not docs_1 or not docs_2:
            return {"error": "One or both documents could not be found or have no indexed content."}

        context_1 = "\n".join(docs_1)
        context_2 = "\n".join(docs_2)

        template = """
        Compare and contrast the following two research documents based on their key findings, methodologies, and conclusions:

        Document 1 ({file_name_1}):
        {context_1}

        Document 2 ({file_name_2}):
        {context_2}

        Comparative Analysis:
        """
        prompt = PromptTemplate(
            template=template, 
            input_variables=["file_name_1", "file_name_2", "context_1", "context_2"]
        )
        formatted = prompt.format(
            file_name_1=file_name_1, 
            file_name_2=file_name_2, 
            context_1=context_1, 
            context_2=context_2
        )
        response = self.llm.invoke(formatted)

        return {
            "document_1": file_name_1,
            "document_2": file_name_2,
            "comparison": response.content
        }