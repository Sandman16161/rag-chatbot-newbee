import gradio as gr
from ingest import load_and_split
from embed import get_embeddings
from vector_store import store_embeddings, load_vector_store
from query import get_qa_chain
import os
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = "data/hr_policies.pdf"

docs = load_and_split(PDF_PATH)
embeddings = get_embeddings()
store_embeddings(docs, embeddings)
vector_store = load_vector_store(embeddings)
qa_chain = get_qa_chain(vector_store)

def answer_question(q):
    return qa_chain.run(q)

iface = gr.Interface(fn=answer_question, inputs="text", outputs="text", title="New Joiner HR Bot")
iface.launch()
