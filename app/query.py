from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
