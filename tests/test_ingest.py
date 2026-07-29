from langchain_core.documents import Document

from src.ingest import chunk_documents, load_documents


def test_load_documents_reads_txt_and_md_but_skips_unsupported(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.md").write_text("# heading\ncontent", encoding="utf-8")
    (tmp_path / "c.json").write_text("{}", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    contents = {doc.page_content for doc in documents}
    assert "hello world" in contents
    assert "# heading\ncontent" in contents


def test_chunk_documents_splits_long_text():
    long_text = "word " * 500
    documents = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    chunks = chunk_documents(documents, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)
