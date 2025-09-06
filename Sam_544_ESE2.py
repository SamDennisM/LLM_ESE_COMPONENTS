import streamlit as st
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt


PERSIST_DIR = "chroma_store"
GROQ_API_KEY = "gsk_wYZZXrWHPP0Ehy3RFpGAu8pMGMbdJOJhmvFLazS"  
LLM_MODEL = "llama-3.1-8b-instant"  


@st.cache_resource
def load_vectorstore(docs_path="F:\LLM\docs"):
    """Load and preprocess domain docs into Chroma DB."""
    documents = []
    for file in os.listdir(docs_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_path, file))
            documents.extend(loader.load())


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb


def get_chat_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(model=LLM_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.3)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory
    )
    return qa_chain



st.set_page_config(page_title="Domain QA Chatbot", layout="wide")
st.title("🤖 Domain-Specific QA Chatbot - Bio")
st.markdown("Chat with your domain documents interactively. Multi-turn, context-aware!")


with st.sidebar:
    st.header("⚙️ Settings")
    docs_folder = st.text_input("Documents folder", "domain_docs")
    if st.button("Load Documents"):
        with st.spinner("Indexing documents..."):
            vectordb = load_vectorstore(docs_folder)
            st.session_state["qa_chain"] = get_chat_chain(vectordb)
        st.success("Documents loaded!")


if "qa_chain" not in st.session_state:
    st.warning("Please load domain documents first from the sidebar.")
else:
    qa_chain = st.session_state["qa_chain"]
    if "messages" not in st.session_state:
        st.session_state.messages = []

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.run(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


if st.session_state.get("messages"):
    st.subheader("📊 Conversation Analysis")

  
    num_turns = len([m for m in st.session_state.messages if m["role"] == "user"])

    
    response_lengths = [len(m["content"].split()) for m in st.session_state.messages if m["role"] == "assistant"]
    avg_resp_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    col1, col2 = st.columns(2)
    col1.metric("Number of Turns", num_turns)
    col2.metric("Avg Response Length", f"{avg_resp_len:.1f} words")

    
    fig, ax = plt.subplots()
    ax.plot(range(1, len(response_lengths) + 1), response_lengths, marker='o')
    ax.set_title("Response Length per Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Words")
    st.pyplot(fig)
