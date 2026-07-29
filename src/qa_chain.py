"""Retrieval-augmented Q&A chain over the Pinecone vector store, with safety guardrails applied."""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.guardrails import EMERGENCY_RESPONSE, SCOPE_GUARD_INSTRUCTIONS, add_disclaimer, check_emergency

DEFAULT_MODEL = "gpt-4o-mini"

QA_PROMPT = ChatPromptTemplate.from_template(
    SCOPE_GUARD_INSTRUCTIONS
    + "\n\nContext:\n{context}\n\nQuestion: {input}"
)


def build_qa_chain(vector_store: PineconeVectorStore, model: str = DEFAULT_MODEL, top_k: int = 4):
    llm = ChatOpenAI(model=model, temperature=0.2)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    combine_docs_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    return create_retrieval_chain(retriever, combine_docs_chain)


def ask(chain, question: str) -> dict:
    """Return {'answer': str, 'sources': list[str], 'is_emergency': bool} for the given question.

    Emergency-flagged questions bypass the RAG/LLM call entirely and return a
    fixed response directing the user to emergency services.
    """
    if check_emergency(question):
        return {"answer": EMERGENCY_RESPONSE, "sources": [], "is_emergency": True}

    result = chain.invoke({"input": question})
    sources = sorted({doc.metadata.get("source", "unknown") for doc in result["context"]})
    return {"answer": add_disclaimer(result["answer"]), "sources": sources, "is_emergency": False}
