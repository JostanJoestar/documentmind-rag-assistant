from __future__ import annotations

import os

import streamlit as st

from rag.embeddings import SentenceTransformerEmbedder
from rag.generation import ExtractiveGenerator, OllamaGenerator, OpenAIGenerator
from rag.pdf import PDFExtractionError
from rag.pipeline import RAGPipeline


st.set_page_config(page_title="DocumentMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading local embedding model …")
def load_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def build_generator(provider: str, model: str, api_key: str):
    if provider == "Ollama (local & free)":
        return OllamaGenerator(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "OpenAI API":
        return OpenAIGenerator(api_key=api_key, model=model)
    return ExtractiveGenerator()


def reset_chat() -> None:
    st.session_state.messages = []


def render_sources(sources) -> None:
    with st.expander(f"View {len(sources)} retrieved sources"):
        for number, source in enumerate(sources, start=1):
            chunk = source.chunk
            st.markdown(
                f"**Source {number} · {chunk.document_name} · "
                f"page {chunk.page_number} · similarity {source.score:.3f}**"
            )
            st.write(chunk.text)
            if number < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📄 DocumentMind")
st.caption("Ask questions about PDFs with local embeddings, semantic search and cited sources.")

with st.sidebar:
    st.header("1 · Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("Retrieval settings"):
        chunk_size = st.slider("Chunk size (words)", 80, 350, 180, 10)
        overlap = st.slider("Overlap (words)", 0, 80, 35, 5)
        top_k = st.slider("Retrieved sources", 1, 8, 4)

    st.header("2 · Answer model")
    provider = st.selectbox(
        "Provider",
        ["Ollama (local & free)", "Retrieval only (no LLM)", "OpenAI API"],
    )

    api_key = ""
    if provider == "Ollama (local & free)":
        model = st.text_input("Ollama model", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
        st.caption("Run `ollama pull llama3.2:3b` once before asking questions.")
    elif provider == "OpenAI API":
        model = st.text_input("OpenAI model", value="gpt-4o-mini")
        api_key = st.text_input("API key", type="password")
        st.caption("The key is kept only in this browser session.")
    else:
        model = "retrieval-only"

    process = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process:
        if overlap >= chunk_size:
            st.error("Overlap must be smaller than chunk size.")
        else:
            try:
                with st.spinner("Extracting text and creating embeddings …"):
                    pipeline = RAGPipeline(
                        embedder=load_embedder(),
                        generator=build_generator(provider, model, api_key),
                    )
                    count = pipeline.ingest(
                        [(file.name, file.getvalue()) for file in uploaded_files],
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    st.session_state.pipeline = pipeline
                    st.session_state.document_names = [file.name for file in uploaded_files]
                    reset_chat()
                st.success(f"Indexed {count} text chunks.")
            except (PDFExtractionError, ValueError, RuntimeError) as exc:
                st.error(str(exc))

    if "pipeline" in st.session_state:
        st.divider()
        st.success("Index ready")
        for name in st.session_state.document_names:
            st.caption(f"• {name}")
        if st.button("Clear index", use_container_width=True):
            del st.session_state.pipeline
            st.session_state.pop("document_names", None)
            reset_chat()
            st.rerun()

if "pipeline" not in st.session_state:
    st.info("Upload a text-based PDF in the sidebar and click **Process documents**.")
    st.markdown(
        """
        **How it works**

        1. Text is extracted page by page.
        2. Overlapping chunks are transformed into local embeddings.
        3. Your question retrieves the semantically closest chunks.
        4. The answer model receives only those chunks and cites its sources.
        """
    )
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving evidence and generating answer …"):
                    pipeline = st.session_state.pipeline
                    pipeline.generator = build_generator(provider, model, api_key)
                    result = pipeline.ask(question, top_k=top_k)
                st.markdown(result.answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
                render_sources(result.sources)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

