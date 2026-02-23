"""RAG utilities for LangChain pipelines."""

import os
from typing import Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq


class RAGHelper:
    """Helper for RAG-related operations."""

    DEFAULT_PROMPT_TEMPLATE = """{context}

Question: {query}

Answer:"""

    @classmethod
    def create_llm(cls, config: dict[str, Any]) -> ChatGroq | None:
        """Create ChatGroq LLM from config.

        Args:
            config: Configuration dictionary.

        Returns:
            ChatGroq instance or None if RAG disabled.
        """
        rag_config = config.get("rag", {})
        if not rag_config.get("enabled", False):
            return None

        model = rag_config.get("model", "llama-3.3-70b-versatile")
        api_key = rag_config.get("api_key") or os.environ.get("GROQ_API_KEY")
        temperature = rag_config.get("temperature", 0.7)
        max_tokens = rag_config.get("max_tokens", 2048)

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def format_prompt(
        cls, query: str, documents: list[Document], template: str | None = None
    ) -> str:
        """Format RAG prompt with query and documents.

        Args:
            query: Query text.
            documents: List of retrieved documents.
            template: Optional custom prompt template.

        Returns:
            Formatted prompt.
        """
        if template is None:
            template = cls.DEFAULT_PROMPT_TEMPLATE

        context = "\n\n".join(
            [
                f"Document {i + 1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )

        return template.format(context=context, query=query)

    @classmethod
    def generate(
        cls,
        llm: ChatGroq,
        query: str,
        documents: list[Document],
        template: str | None = None,
    ) -> str:
        """Generate RAG answer using LLM.

        Args:
            llm: ChatGroq instance.
            query: Query text.
            documents: List of retrieved documents.
            template: Optional custom prompt template.

        Returns:
            Generated answer.
        """
        prompt = cls.format_prompt(query, documents, template)
        response = llm.invoke(prompt)
        return response.content
