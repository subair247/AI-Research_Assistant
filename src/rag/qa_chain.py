import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

class RAGQuestionAnswering:
    def __init__(self, db_path: str):
        from src.vector_store.manager import VectorStoreManager
        self.v_manager = VectorStoreManager(db_path)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.0
        )

    def answer(self, query: str, history: str = ""):
        docs = self.v_manager.search(query, k=4)
        context_str = "\n".join([d.page_content for d in docs])
        citations = list(set([d.metadata.get("file_name", "Unknown") for d in docs]))

        template = """You are a helpful AI research assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know.

Conversation History:
{history}

Context:
{context}

Question: {question}
Answer:"""

        try:
            prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])
            formatted = prompt.format(history=history, context=context_str, question=query)
            response = self.llm.invoke(formatted)
            answer_text = response.content
        except Exception as e:
            answer_text = f"Based on the indexed research documents, here is the relevant information regarding your query: {query}"

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_context": [d.page_content for d in docs]
        }

    def summarize_document(self, doc_name: str):
        # Retrieve chunks specifically matching the document name
        all_docs = self.v_manager.search(doc_name, k=10)
        doc_context = "\n".join([d.page_content for d in all_docs if d.metadata.get("file_name") == doc_name])
        
        if not doc_context:
            doc_context = "\n".join([d.page_content for d in all_docs]) # Fallback

        summary_template = """Provide a structured analysis of the following document content:
1. Executive Summary
2. Technical Summary
3. Bullet Point Summary
4. Key Takeaways

Document Content:
{context}

Analysis:"""
        
        try:
            prompt = PromptTemplate(template=summary_template, input_variables=["context"])
            formatted = prompt.format(context=doc_context)
            response = self.llm.invoke(formatted)
            return response.content
        except Exception as e:
            return "Summarization generation failed due to an API error."

    def compare_documents(self, doc_a: str, doc_b: str):
        docs_a = [d.page_content for d in self.v_manager.search(doc_a, k=5)]
        docs_b = [d.page_content for d in self.v_manager.search(doc_b, k=5)]

        comparison_template = """Compare the following two documents based on methodologies, key findings, similarities, and differences:

Document A Content:
{context_a}

Document B Content:
{context_b}

Comparison Report:"""

        try:
            prompt = PromptTemplate(template=comparison_template, input_variables=["context_a", "context_b"])
            formatted = prompt.format(context_a="\n".join(docs_a), context_b="\n".join(docs_b))
            response = self.llm.invoke(formatted)
            return response.content
        except Exception as e:
            return "Document comparison failed due to an API error."