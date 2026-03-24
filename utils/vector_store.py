"""
FAISS Vector Store utilities.
"""

import os
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_FAISS_DIR = "./faiss_index"


def _try_st_cache(func):
    try:
        import streamlit as st
        return st.cache_resource(func)
    except Exception:
        return func


def get_faiss_store_path() -> str:
    os.makedirs(_FAISS_DIR, exist_ok=True)
    return _FAISS_DIR


get_faiss_store_path = _try_st_cache(get_faiss_store_path)


def get_vector_store(embeddings: Embeddings, collection_name: str = "rag_collection"):
    index_path = get_faiss_store_path()
    index_file = os.path.join(index_path, f"{collection_name}.faiss")

    if os.path.exists(index_file):
        try:
            return FAISS.load_local(
                index_path,
                embeddings,
                collection_name,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning("Failed to load FAISS index: %s", e)
            return None

    return None


def save_vector_store(vector_store: FAISS, collection_name: str = "rag_collection"):
    index_path = get_faiss_store_path()
    vector_store.save_local(index_path, collection_name)
    logger.info("FAISS index saved to %s", index_path)
