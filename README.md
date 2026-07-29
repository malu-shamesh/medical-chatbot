# medical-chatbot

Integrated LLMs, Pinecone, and LangChain for medical context retrieval, showcasing multi modal Generative AI capabilities.

**This chatbot provides general medical information only. It is not a substitute for professional medical advice, diagnosis, or treatment, and it is not equipped to handle emergencies.**

This repo previously contained no code - only this README describing the intended project. Built out from scratch as a runnable RAG + vision pipeline rather than a converted notebook (there was nothing to convert), with guardrails treated as a first-class part of the design given the domain.

## Pipeline

1. **Ingestion** ([src/ingest.py](src/ingest.py)) - loads `.txt`, `.md`, and `.pdf` medical reference documents from a directory and splits them into overlapping chunks.
2. **Vector store** ([src/vectorstore.py](src/vectorstore.py)) - embeds chunks with OpenAI's `text-embedding-3-small` and upserts them into a Pinecone serverless index (creating the index if it doesn't exist).
3. **Text Q&A chain** ([src/qa_chain.py](src/qa_chain.py)) - retrieves the top-k most relevant chunks and answers with `gpt-4o-mini`, scoped to general information only (see **Guardrails**).
4. **Image Q&A** ([src/vision.py](src/vision.py)) - answers questions about a medical image (e.g. a photo of a rash) using a vision-capable LLM, restricted to describing general visible characteristics rather than diagnosing.
5. **Guardrails** ([src/guardrails.py](src/guardrails.py)) - shared across both text and image paths.

## Guardrails

A medical chatbot carries real safety and liability weight a generic RAG assistant doesn't. This is enforced in code, not just prompt wording, because prompt instructions can be missed or worked around but a keyword check on the input cannot:

- **Emergency detection**: user input is checked against a list of red-flag symptom/crisis patterns (chest pain, difficulty breathing, suicidal ideation, overdose, anaphylaxis, etc.) *before* any RAG or LLM call. A match short-circuits straight to a fixed response directing the user to emergency services (911 / 988 in the US) - it never reaches the LLM, so there's no chance of the model reasoning its way past the guardrail on an emergency input.
- **Scope guard**: the system prompt instructs the model to describe only general information found in retrieved context, never to diagnose, prescribe, or give personalized dosing/treatment advice, and to say so plainly when the context doesn't contain an answer. The image-analysis prompt adds an explicit reminder against diagnosis-shaped language, since vision models are especially prone to "this looks like..." phrasing when shown a medical photo.
- **Disclaimer**: appended to every non-emergency response, so the informational (not diagnostic) nature of the tool is never left implicit.

These are necessary, not sufficient - see **Limitations** below.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` and `PINECONE_API_KEY` (via `.env` or your shell environment).

## Usage

```bash
# 1. Place .txt/.md/.pdf medical reference files in docs/, then ingest them
python main.py ingest --docs-dir docs

# 2. Ask a text question
python main.py ask --question "What are common symptoms of the flu?"

# 3. Ask about an image
python main.py ask-image --image rash.jpg --question "What might cause this?"

# 4. Or start an interactive chat loop
python main.py chat
```

## Tests

```bash
pytest
```

`tests/test_guardrails.py` is the most important test file here - it exercises the emergency-detection patterns against both example emergency phrasings and routine medical questions, to catch both false negatives (missing a real emergency) and false positives (needlessly blocking a normal question).

## Limitations

- **Emergency detection is keyword-based and not exhaustive.** It will not catch every way a person might describe an emergency, especially in another language, with typos, or indirectly. It is a safety floor, not a complete triage system - real deployments should pair this with human oversight and should not be the sole safety mechanism.
- **The scope guard depends on the LLM following instructions.** LLMs can still occasionally produce diagnosis-shaped language despite being told not to. This reduces but does not eliminate that risk.
- **Retrieval quality depends entirely on the ingested documents.** If `docs/` doesn't contain accurate, appropriately-licensed medical reference material, the chatbot's "general information" is only as good as what was ingested.

## Verification note

This machine's Python (3.14) hit a real `langchain`/`pydantic` incompatibility while converting a related repo in this same session (`conversational-ai`) - `langchain.chains`, which `src/qa_chain.py` also imports, fails with a `typing.Optional` forward-reference evaluation error specific to this Python version. There is also no `OPENAI_API_KEY` or `PINECONE_API_KEY` available in this environment. `src/guardrails.py` and `src/ingest.py` (the two modules with no such dependency) were verified locally with passing tests, including the emergency-detection test suite. Verify a full `ingest`/`ask`/`ask-image` run - ideally on Python 3.11/3.12 with real API keys - before relying on this code, and have a qualified reviewer sign off on the guardrail behavior before any real-world use.
