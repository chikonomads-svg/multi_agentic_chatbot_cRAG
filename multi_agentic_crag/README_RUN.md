How to run the Multi-Agentic CRAG Streamlit app (local)

1. Create a virtualenv and install requirements:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r multi_agentic_crag/requirements.txt

2. Copy multi_agentic_crag/.env.example to .env at the repository root and fill
   in your Azure & Tavily keys. Example variables used:

   AZURE_OPENAI_API_KEY
   AZURE_OPENAI_ENDPOINT
   AZURE_OPENAI_API_VERSION
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
   TAVILY_API_KEY

3. Run Streamlit:

   streamlit run multi_agentic_crag/src/app.py

Notes:
- The app uses Azure OpenAI when AZURE_* env vars are present. It maps
  those to OPENAI_* environment variables via src.utils.env.load_env().
- Use the sidebar to build or load FAISS indices. Built indices are
  saved under data/vector_dbs/<index_name> (index.faiss, index.pkl).
