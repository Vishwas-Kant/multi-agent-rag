import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document
from utils.llm import get_embeddings
from utils.vector_store import get_vector_store, save_vector_store
from utils.cache import cached

_retriever = None


def extract_text_from_pdf(pdf_paths: list) -> list:
    try:
        import pymupdf4llm

        all_docs = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            md_text = pymupdf4llm.to_markdown(pdf_path)

            if md_text.strip():
                all_docs.append(Document(
                    page_content=md_text,
                    metadata={
                        "source": pdf_path,
                        "format": "markdown",
                        "extraction_method": "pymupdf4llm"
                    }
                ))

        return all_docs

    except (ImportError, Exception):
        all_docs = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

        return all_docs


def initialize_rag(pdf_paths: list):
    global _retriever

    all_docs = extract_text_from_pdf(pdf_paths)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        add_start_index=True,
        separators=["\n\n", "\n", ".", "•", " ", ""],
    )
    splits = text_splitter.split_documents(all_docs)

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if vector_store is None:
        from langchain_community.vectorstores import FAISS
        vector_store = FAISS.from_documents(splits, embeddings)
    else:
        vector_store.add_documents(documents=splits)

    save_vector_store(vector_store)

    _retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    if hasattr(retrieve_context_cached, "cache"):
        retrieve_context_cached.cache.clear()

    extraction_method = all_docs[0].metadata.get('extraction_method', 'pypdf') if all_docs else 'pypdf'
    return f"RAG system initialized with {len(pdf_paths)} file(s) and {len(splits)} chunks [{extraction_method.upper()}]"


@cached(ttl=120)
def retrieve_context_cached(query: str) -> str:
    global _retriever
    if _retriever is None:
        return "Error: RAG system not initialized. Please upload a PDF first."

    docs = _retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def retrieve_context(query: str) -> str:
    return retrieve_context_cached(query)
