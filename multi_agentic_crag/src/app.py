"""Streamlit app entry point for Multi-Agent Corrective RAG."""
from typing import Optional, List
import streamlit as st
import os
import logging
import json

import sys
from pathlib import Path

# When streamlit runs this file as a script the package relative imports fail;
# ensure the package root (multi_agentic_crag) is on sys.path and import via
# the `src` package to keep imports stable both when running as a module and
# via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph import build_graph, State
from src.utils.env import load_env
from src.utils.loaders import load_documents_from_dir
from src.vector_store import create_and_persist_faiss, load_faiss, process_folder
import shutil
from src.agents.analyst import detect_and_build_plot
from src.logging_config import configure_logging

configure_logging()

LOG = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Multi-Agentic CRAG")
    # Load environment variables from parent .env if present (keeps keys out of repo)
    load_env()
    st.sidebar.title("Settings")
    # Load keys silently from the environment (.env); do NOT show or ask for
    # them in the UI. Ensure your parent .env contains AZURE_OPENAI_API_KEY or
    # OPENAI_API_KEY and TAVILY_API_KEY before running the app.
    tavily = os.getenv("TAVILY_API_KEY", "")
    st.sidebar.markdown("---")

    index_action = st.sidebar.radio("Index", ["Load Existing Index", "Build New Index"]) 
    index_name = st.sidebar.text_input("Index name", value="my_vector_db")

    if index_action == "Build New Index":
        upload_dir = st.sidebar.text_input("Directory to ingest (local)")
        uploaded_files = st.sidebar.file_uploader("Upload files to ingest", accept_multiple_files=True)

        # If the user uploaded files via the UI, save them into a local
        # directory and use that directory for ingestion. Otherwise fall back
        # to the provided local directory path.
        files_dir = None
        if uploaded_files:
            files_dir = os.path.join("data", "uploads", index_name)
            os.makedirs(files_dir, exist_ok=True)
            for f in uploaded_files:
                # Streamlit's UploadedFile provides getbuffer() to get bytes
                target = os.path.join(files_dir, f.name)
                with open(target, "wb") as out:
                    out.write(f.getbuffer())

        ingest_path = files_dir or upload_dir
        if st.sidebar.button("Ingest and Build"):
            if not ingest_path:
                st.sidebar.error("Please provide a directory or upload files to ingest")
            else:
                # If the path is a directory, use the parallel folder ingestion
                # which will create an index under <folder>/index. We'll then
                # copy the persisted index into the app's data/vector_dbs
                # directory so it appears in the UI like other indices.
                if os.path.isdir(ingest_path):
                    st.sidebar.info("Starting parallel folder ingestion (may take some time)...")
                    vs = process_folder(ingest_path, source=index_name)
                    if vs is not None:
                        derived = os.path.join(ingest_path, "index")
                        dest = os.path.join("data", "vector_dbs", index_name)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        try:
                            # copytree with dirs_exist_ok ensures we overwrite/update
                            shutil.copytree(derived, dest, dirs_exist_ok=True)
                        except Exception:
                            # fallback: attempt to copy files individually
                            try:
                                for root, _, files in os.walk(derived):
                                    rel = os.path.relpath(root, derived)
                                    target_dir = os.path.join(dest, rel)
                                    os.makedirs(target_dir, exist_ok=True)
                                    for f in files:
                                        shutil.copy2(os.path.join(root, f), os.path.join(target_dir, f))
                            except Exception:
                                st.sidebar.warning("Index created but failed to copy into data/vector_dbs; check logs")
                        st.sidebar.success(f"Index built and copied to data/vector_dbs/{index_name}")
                    else:
                        st.sidebar.error("Folder ingestion failed; see logs")
                else:
                    # treat ingest_path as a single-folder of files to load
                    docs = load_documents_from_dir(ingest_path)
                    # create_and_persist_faiss will use Azure embeddings if configured
                    idx = create_and_persist_faiss(docs, f"data/vector_dbs/{index_name}")
                    st.sidebar.success("Index built")

    if index_action == "Load Existing Index":
        # show available indices
        # Ensure the vector_dbs directory exists before listing. Streamlit runs
        # may start in an environment where data/vector_dbs hasn't been created
        # yet (e.g. a fresh checkout). Create the directory if missing and
        # fall back to an empty list so the UI doesn't raise FileNotFoundError.
        db_base = os.path.join("data", "vector_dbs")
        try:
            os.makedirs(db_base, exist_ok=True)
            available = [d for d in os.listdir(db_base) if os.path.isdir(os.path.join(db_base, d))]
        except Exception:
            LOG.exception("Failed to access or create data/vector_dbs directory")
            st.sidebar.error("Unable to access data/vector_dbs; check application logs")
            available = []

        sel = st.sidebar.selectbox("Available indices", options=[""] + available, index=0)
        if sel:
            index_name = sel
        if st.sidebar.button("Load Index"):
            idx = load_faiss(f"data/vector_dbs/{index_name}")
            if idx:
                st.sidebar.success("Index loaded")
            else:
                st.sidebar.warning("Index not found or failed to load")

    st.header("Multi-Agent Corrective RAG Chat")
    query = st.text_input("Ask a question")
    if st.button("Submit") and query:
        # Very simple orchestration: router -> retriever -> grader -> fallback -> generator
        state: State = {"user_message": query, "faiss_index": f"data/vector_dbs/{index_name}", "tavily_api_key": tavily}
        g = build_graph()
        decision = g.run_node("router", state)
        st.write(f"Routed to: {decision}")
        if decision == "local":
            retrieved = g.run_node("retriever", state)
            score = g.run_node("grader", state, retrieved)
            if score < 0.5:
                st.write("Grader low — falling back to web search")
                web = g.run_node("web_search", state)
                # run generator and handle both streaming and non-streaming
                stream_gen = g.run_node("generator", state, [str(web)])
                if hasattr(stream_gen, "__iter__") and not isinstance(stream_gen, (str, bytes)):
                    for chunk in stream_gen:
                        st.write(chunk)
                    answer = ""
                else:
                    st.write(stream_gen)
                    answer = ""
            else:
                # build sources list (use top-2 most relevant chunks for grounding)
                top_k = 2
                top = retrieved[:top_k] if isinstance(retrieved, list) else retrieved
                sources = []
                # include simple citation info if available in metadata
                for r in top:
                    doc = r[0]
                    meta = getattr(doc, 'metadata', {}) or {}
                    source_name = meta.get('source') or meta.get('source_name') or ''
                    if source_name:
                        sources.append(f"Source: {source_name}\n{doc.page_content}")
                    else:
                        sources.append(doc.page_content)
                # detect plot
                plot_json = detect_and_build_plot("\n".join(sources))
                if plot_json:
                    # convert to a simple plotly figure when applicable
                    try:
                        import plotly.graph_objs as go
                        fig = None
                        if plot_json.get("type") == "bar":
                            fig = go.Figure(go.Bar(x=plot_json["data"]["x"], y=plot_json["data"]["y"]))
                        elif plot_json.get("type") == "line":
                            fig = go.Figure(go.Line(x=plot_json["data"]["x"], y=plot_json["data"]["y"]))
                        if fig is not None:
                            st.plotly_chart(fig)
                    except Exception:
                        st.write("(failed to render plot)")
                # stream generation from local sources
                stream_gen = g.run_node("generator", state, sources)
                # generator may expose a streaming generator; detect that
                if hasattr(stream_gen, "__iter__") and not isinstance(stream_gen, str):
                    answer = ""
                    for chunk in stream_gen:
                        st.write(chunk)
                else:
                    answer = stream_gen
        else:
            web = g.run_node("web_search", state)
            # stream web-based generation
            stream_gen = g.run_node("generator", state, [str(web)])
            if hasattr(stream_gen, "__iter__") and not isinstance(stream_gen, (str, bytes)):
                for chunk in stream_gen:
                    st.write(chunk)
                answer = ""
            else:
                answer = stream_gen

        if answer:
            st.write(answer)


if __name__ == "__main__":
    main()
