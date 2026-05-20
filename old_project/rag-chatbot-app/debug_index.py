import logging
import os
import traceback

logging.basicConfig(level=logging.DEBUG)

from src.processor import load_documents, chunk_documents
from src.vector_store import create_vector_db, get_vector_db_path

def main():
    src = 'rag-chatbot-app/data/agentic_ai_resource.pdf'
    print('Loading document:', src)
    docs = load_documents([src])
    print('Loaded docs count:', len(docs))
    chunks = chunk_documents(docs)
    print('Chunks count:', len(chunks))
    try:
        idx = create_vector_db(chunks, 'my_vector_db2')
        print('create_vector_db returned:', type(idx))
    except Exception:
        print('create_vector_db raised exception:')
        traceback.print_exc()

    p = get_vector_db_path('my_vector_db2')
    print('Vector DB path:', p)
    for fn in ['index.faiss', 'index.pkl']:
        path = os.path.join(p, fn)
        print(fn, 'exists:', os.path.exists(path), 'size:', os.path.getsize(path) if os.path.exists(path) else 'N/A')

if __name__ == '__main__':
    main()
