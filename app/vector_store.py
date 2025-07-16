from langchain.vectorstores import FAISS

def create_vector_store(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

def save_vector_store(vs, path="faiss_index"):
    vs.save_local(path)

def load_vector_store(embeddings, path="faiss_index"):
    return FAISS.load_local(path, embeddings)
