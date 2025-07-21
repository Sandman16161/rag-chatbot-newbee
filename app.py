import os
import time
import re
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler

# ✅ Load OpenAI API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Please set the OPENAI_API_KEY environment variable.")

# ✅ Initialize OpenAI Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Load or Build FAISS Index
INDEX_DIR = "faiss_index"
if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    db = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
else:
    print("⚠️ FAISS index not found. Creating from PDF...")

    # ✅ List your pre-uploaded PDFs
    datar = ["admin_guide.pdf", "hr_policies.pdf", "it_help_manual.pdf", "server_issues_guide.pdf"]

    docs = []
    for pdf in datar:
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        docs.extend(pages)

    # ✅ Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # ✅ Embed and save the FAISS index
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(INDEX_DIR)

# ✅ Create Retriever
retriever = db.as_retriever()

# ✅ Token Streaming Handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.output = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.output += token

# ✅ Main QA Logic
def gradio_streaming_answer(query, chat_history):
    handler = StreamHandler()
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[handler],
        temperature=0.1,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    answer = handler.output or result["result"]

    # Format into bullet steps if possible
    steps = re.split(r"(?=\d+\.\s)", answer.strip())
    formatted_answer = "\n\n".join(step.strip() for step in steps)

    sources = result.get("source_documents", [])
    formatted_sources = "\n".join([
        f"- 📄 {doc.metadata.get('source', 'Unknown')} – Page {doc.metadata.get('page', 'N/A')}"
        for doc in sources
    ])

    full_response = f"{formatted_answer}\n\n**📚 Sources:**\n{formatted_sources}"

    chat_history.append((query, ""))
    streamed = ""
    for line in full_response.split("\n"):
        streamed += line + "\n"
        chat_history[-1] = (query, streamed.strip())
        yield chat_history, chat_history
        time.sleep(0.1)

# ✅ Gradio UI
with gr.Blocks() as iface:
    gr.Markdown("## 🧠 RAG Chatbot\nAsk questions specific to HR,Admin,IT, and service issues")

    chatbot = gr.Chatbot(
        label="Your AI Assistant",
        show_copy_button=True,
        avatar_images=["user_avatar.jpg", "bot_avatar.jpg"],
        layout="bubble"
    )
    query_input = gr.Textbox(placeholder="Ask your question...")
    clear = gr.Button("🔁 Clear Chat")

    state = gr.State([])

    query_input.submit(gradio_streaming_answer, [query_input, state], [chatbot, state])
    query_input.submit(lambda: "", None, query_input)

    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)

iface.launch()
