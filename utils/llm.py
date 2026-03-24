"""
LLM & Embedding model initialization utilities.
"""

import os
import logging

logger = logging.getLogger(__name__)

_llm_cache = None
_embeddings_cache = None


def _try_st_cache(func):
    try:
        import streamlit as st
        return st.cache_resource(func)
    except Exception:
        return func


def get_llm():
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    from langchain_community.chat_models import ChatLlamaCpp
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    from huggingface_hub import hf_hub_download

    repo_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    filenames = [
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
    ]

    model_path = None
    for filename in filenames:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        if filename.endswith("00001-of-00002.gguf"):
            model_path = path

    verbose = os.getenv("LLM_VERBOSE", "false").lower() == "true"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]) if verbose else None

    llm = ChatLlamaCpp(
        model_path=model_path,
        temperature=0.1,
        max_tokens=2000,
        n_ctx=4096,
        callback_manager=callback_manager,
        verbose=verbose,
    )

    _llm_cache = llm
    return llm


def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    from langchain_huggingface import HuggingFaceEmbeddings

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    model_kwargs = {
        "device": "cpu",
        "token": hf_token,
    }

    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    _embeddings_cache = embeddings
    return embeddings

get_llm = _try_st_cache(get_llm)
get_embeddings = _try_st_cache(get_embeddings)
