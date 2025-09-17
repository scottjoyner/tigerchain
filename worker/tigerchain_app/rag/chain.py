from __future__ import annotations

from typing import Dict, List

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from ..config import Settings
from ..utils.logging import get_logger
from .retriever import TigerGraphVectorRetriever

logger = get_logger(__name__)


QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the provided context.\n\nIf the context does not contain the answer, respond that the answer was not found.\nAlways cite the source filename when available."""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "{system_prompt}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


def build_qa_chain(
    settings: Settings,
    retriever: TigerGraphVectorRetriever,
    llm: BaseLanguageModel,
) -> RetrievalQA:
    """Construct a RetrievalQA chain with TigerGraph as the retriever."""

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_PROMPT.partial(system_prompt=QA_SYSTEM_PROMPT),
        },
    )
    return chain


def format_sources(documents: List) -> List[Dict[str, str]]:
    formatted = []
    for doc in documents:
        metadata = doc.metadata or {}
        formatted.append(
            {
                "title": metadata.get("title"),
                "source": metadata.get("source"),
                "uri": metadata.get("uri"),
                "http_url": metadata.get("http_url"),
                "score": metadata.get("score"),
            }
        )
    return formatted
