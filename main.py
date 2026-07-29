"""CLI for the medical information chatbot (text RAG + image Q&A).

This tool provides general medical information only and is not a substitute
for professional medical advice, diagnosis, or treatment.

Usage:
    python main.py ingest --docs-dir docs
    python main.py ask --question "What are common symptoms of the flu?"
    python main.py ask-image --image rash.jpg --question "What might this be?"
    python main.py chat
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.ingest import chunk_documents, load_documents
from src.qa_chain import ask, build_qa_chain
from src.vectorstore import DEFAULT_INDEX_NAME, build_vector_store, load_vector_store
from src.vision import ask_about_image


def cmd_ingest(args: argparse.Namespace) -> None:
    print(f"Loading documents from {args.docs_dir}...")
    documents = load_documents(args.docs_dir)
    print(f"Loaded {len(documents)} document(s).")

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Split into {len(chunks)} chunk(s).")

    build_vector_store(chunks, args.index_name)
    print(f"Upserted chunks into Pinecone index '{args.index_name}'.")


def _print_result(result: dict) -> None:
    print(f"\n{result['answer']}")
    if result.get("sources"):
        print(f"\nSources: {', '.join(result['sources'])}")


def cmd_ask(args: argparse.Namespace) -> None:
    vector_store = load_vector_store(args.index_name)
    chain = build_qa_chain(vector_store, model=args.model)
    _print_result(ask(chain, args.question))


def cmd_ask_image(args: argparse.Namespace) -> None:
    result = ask_about_image(args.image, args.question, model=args.model)
    _print_result(result)


def cmd_chat(args: argparse.Namespace) -> None:
    vector_store = load_vector_store(args.index_name)
    chain = build_qa_chain(vector_store, model=args.model)

    print("Medical information chatbot. This is not a substitute for professional medical advice.")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = ask(chain, question)
        _print_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medical information chatbot (text RAG + image Q&A)")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--model", default="gpt-4o-mini")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load, chunk, and embed medical reference documents")
    ingest_parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    ingest_parser.add_argument("--chunk-size", type=int, default=1000)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a single text question")
    ask_parser.add_argument("--question", required=True)
    ask_parser.set_defaults(func=cmd_ask)

    ask_image_parser = subparsers.add_parser("ask-image", help="Ask a question about a medical image")
    ask_image_parser.add_argument("--image", type=Path, required=True)
    ask_image_parser.add_argument("--question", required=True)
    ask_image_parser.set_defaults(func=cmd_ask_image)

    chat_parser = subparsers.add_parser("chat", help="Interactive text Q&A loop")
    chat_parser.set_defaults(func=cmd_chat)

    return parser


def main() -> None:
    load_dotenv(override=True)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
