import pytest
from unittest.mock import MagicMock, patch
from src.rag.qa_chain import RAGQuestionAnswering

@patch("src.rag.qa_chain.VectorStoreManager")
@patch("src.rag.qa_chain.ChatOpenAI")
def test_rag_answer_structure(mock_chat_openai, mock_vector_store):
    mock_doc = MagicMock()
    mock_doc.page_content = "This is test research context text."
    mock_doc.metadata = {"file_name": "sample.pdf", "page_number": 1}
    
    mock_instance = mock_vector_store.return_value
    mock_instance.search.return_value = [mock_doc]
    
    mock_llm_instance = mock_chat_openai.return_value
    mock_llm_instance.invoke.return_value = MagicMock(content="Mocked RAG response based on text.")

    rag = RAGQuestionAnswering(vector_db_dir="./data/vector_db")
    result = rag.answer("What is the test research about?", history="")

    assert "answer" in result
    assert "citations" in result
    assert result["citations"][0]["document"] == "sample.pdf"
    assert result["citations"][0]["page"] == 1