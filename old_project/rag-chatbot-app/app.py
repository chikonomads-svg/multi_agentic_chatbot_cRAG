"""Streamlit app entrypoint for the RAG chatbot.

This module performs a runtime dependency check and offers an in-app
installer before importing heavy dependencies (like langchain).
"""
import os
import importlib
import importlib.util
import subprocess
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


LOG = logging.getLogger("rag_chatbot")
logging.basicConfig(level=logging.INFO)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR = DATA_DIR / "faiss_index"

# Load environment variables from a .env file if present so Azure/OpenAI
# settings in the repository root are honored at runtime.
load_dotenv()


REQUIRED_MODULES = [
    "langchain",
    "faiss",
    "streamlit",
    "tavily",
    "plotly",
    "pandas",
    "pypdf",
    "docx2txt",
    "openai",
]


def check_missing_modules():
    missing = []
    for mod in REQUIRED_MODULES:
        if importlib.util.find_spec(mod) is None:
            missing.append(mod)
    return missing


def install_requirements(output_lines: list):
    """Install requirements from requirements.txt in the project root.

    Appends output to output_lines list to show in the UI.
    """
    req_path = Path("requirements.txt")
    if not req_path.exists():
        output_lines.append(f"requirements.txt not found at {req_path.resolve()}")
        return
    cmd = ["python", "-m", "pip", "install", "-r", str(req_path)]
    output_lines.append("Running: %s" % " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        output_lines.append(res.stdout)
        if res.returncode != 0:
            output_lines.append(res.stderr)
    except Exception as e:
        output_lines.append(str(e))


def main():
    st.set_page_config(page_title="RAG Chatbot")
    st.sidebar.title("Controls")

    missing = check_missing_modules()
    if missing:
        st.warning(f"Missing dependencies detected: {missing}")
        st.info("Click the button below to install dependencies into the active Python environment. You may need to restart the app after installation.")
        output_lines = []
        if st.sidebar.button("Install dependencies"):
            install_requirements(output_lines)
        st.text_area("Installer output", value="\n".join(output_lines), height=200)
        # Don't proceed until user resolves deps
        return

    # Now that dependencies exist, import the local modules
    from src.processor import load_documents, chunk_documents
    from src.vector_store import create_faiss_index, load_faiss_index, index_exists
    from src.chains import retrieve_answer
    from src.utils import extract_markdown_table, plot_from_dataframe

    # Sidebar workflow: show available named vector DBs and allow the user to
    # select one, or upload & create a new named vector DB. Provide an option
    # to hide the embedding configuration menu. Also add a small color theme
    # via CSS for a friendlier UI.
    from src.vector_store import list_vector_dbs, create_vector_db, get_vector_db_path, _has_faiss_files, HAS_LANGCHAIN

    # Simple inline CSS to add color accents
    st.markdown(
        """
        <style>
        .reportview-container .main h1 { color: #2A9D8F; }
        .stButton>button { background-color: #264653; color: white; }
        .stSidebar { background-color: #F0F4F8; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    available_dbs = list_vector_dbs()
    st.sidebar.markdown("### Vector DBs")
    # Option to auto-load an existing FAISS index if found on disk
    load_existing_if_present = st.sidebar.checkbox("Load existing FAISS index if present", value=True)

    # If LangChain/FAISS aren't installed, show a prominent warning so the
    # user understands the app will use a lightweight fallback (no real
    # vector similarity) until the dependencies are installed.
    if not HAS_LANGCHAIN:
        st.sidebar.warning(
            "LangChain/FAISS are not installed. App is running in fallback mode — indexing and similarity search will be limited.\n"
            "Install 'langchain' and 'faiss' in your environment to enable full FAISS vector DB functionality."
        )

    selected_db = None
    if available_dbs:
        # Show which DBs already have a persisted FAISS index
        options = ["-- choose --"] + [
            f"{d} (index present)" if _has_faiss_files(get_vector_db_path(d)) else d for d in available_dbs
        ]
        sel = st.sidebar.selectbox("Select a vector DB to use:", options)
        # Normalize selection back to DB name if user selected the annotated option
        if sel and sel != "-- choose --":
            selected_db = sel.split(" ")[0]
    hide_embedding_menu = st.sidebar.checkbox("Hide embedding options", value=True)

    # Embedding options (shown only if not hidden). Only allow Azure/OpenAI
    # configuration - remove local HuggingFace options to avoid confusion.
    if not hide_embedding_menu:
        st.sidebar.markdown("### Embedding configuration (Azure/OpenAI only)")
        st.sidebar.text_input("AZURE_OPENAI_ENDPOINT", value=os.getenv("AZURE_OPENAI_ENDPOINT") or "")
        st.sidebar.text_input("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", value=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "")
        # Allow pasting an Azure key for the current session (optional).
        session_azure_key = st.sidebar.text_input("AZURE_OPENAI_KEY (paste for this session)", value="", type="password")

    # Decide whether to use an existing index or prompt the user to upload/build
    mode = "Upload & build new index"
    # If a selected DB has on-disk FAISS files and the user wants to load existing,
    # switch into use mode automatically.
    if selected_db and load_existing_if_present and _has_faiss_files(get_vector_db_path(selected_db)):
        mode = "Use existing index"
    # Also allow using a default bare INDEX_DIR if it contains a FAISS index
    if not selected_db and load_existing_if_present and _has_faiss_files(str(INDEX_DIR)):
        mode = "Use existing index"

    uploaded = None
    # If the user chose to upload/build, show the uploader in the sidebar
    if mode == "Upload & build new index":
        uploaded = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True)
        db_name = st.sidebar.text_input("Vector DB name", value="my_vector_db")
        if uploaded:
            files = [f.name for f in uploaded]
            # Save uploaded files to the data dir for processing
            saved_paths = []
            for f in uploaded:
                p = DATA_DIR / f.name
                with open(p, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(str(p))

            LOG.info("Loaded %d uploaded files", len(saved_paths))
            docs = load_documents(saved_paths)
            chunks = chunk_documents(docs)
            if st.sidebar.button("Build Index"):
                # If the DB already exists, ask for overwrite confirmation
                existing = get_vector_db_path(db_name) if db_name else None
                exists = os.path.exists(existing) if existing else False
                do_create = True
                if exists:
                    # If a persisted FAISS index already exists and the user chose to
                    # load existing indexes, offer to load it instead of rebuilding.
                    if _has_faiss_files(existing) and load_existing_if_present:
                        load_instead = st.sidebar.checkbox("Load existing FAISS index instead of rebuilding?", value=True)
                        if load_instead:
                            st.sidebar.info("Will load existing FAISS index for this DB.")
                            selected_db = db_name
                            mode = "Use existing index"
                            do_create = False
                    else:
                        confirm = st.sidebar.checkbox("Overwrite existing vector DB?", value=False)
                        if not confirm:
                            st.sidebar.warning("Check 'Overwrite existing vector DB?' to proceed")
                            do_create = False

                if do_create:
                    # If user provided an Azure key in the sidebar, use it for this run
                    if not hide_embedding_menu and session_azure_key:
                        os.environ["AZURE_OPENAI_KEY"] = session_azure_key
                        # Also set OPENAI_ envs so langchain/OpenAIEmbeddings pick it up
                        os.environ["OPENAI_API_KEY"] = session_azure_key
                    # Show a progress indicator while embeddings/index are built
                    with st.spinner("Building index and computing embeddings..."):
                        create_vector_db(chunks, db_name)
                    st.sidebar.success(f"Index '{db_name}' created")
                    selected_db = db_name
                    mode = "Use existing index"

    st.title("RAG Chatbot")
    # Only show the query UI when the user selected to use the existing index
    # (either because one was already on disk, or because they just built one).
    if mode == "Use existing index":
        st.info("Index is available — enter a query below to retrieve an answer.")
        query = st.text_input("Ask a question:")
        if st.button("Submit") and query:
            # Choose faiss path: if a named DB was selected use it, otherwise
            # fall back to the default INDEX_DIR.
            if selected_db and selected_db != "-- choose --":
                faiss_path = get_vector_db_path(selected_db)
            else:
                faiss_path = str(INDEX_DIR)

            answer, sources = retrieve_answer(query, faiss_path, tavily_api_key=os.getenv("TAVILY_API_KEY"))

            # Display answer in a concise chatbot style
            st.markdown("**Assistant:**")
            # Use a blockquote for a chat-like look
            st.markdown(f"> {answer}")

            # If the assistant returned a markdown table, render it as a dataframe
            # so the user sees a proper table. Otherwise just show the answer.
            try:
                df = extract_markdown_table(answer)
                st.markdown("**Answer (table):**")
                st.dataframe(df)
                # Also show a simple plot if the table has numeric columns
                try:
                    fig = plot_from_dataframe(df)
                    st.plotly_chart(fig)
                except Exception:
                    LOG.debug("Table has no numeric columns to plot")
            except Exception:
                # No table found; show the answer text only
                st.markdown("**Answer:**")
                st.write(answer)
    else:
        st.info("Please upload documents in the left sidebar and click 'Build Index' to enable querying.")


if __name__ == "__main__":
    main()
