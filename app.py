import streamlit as st
from streamlit.logger import get_logger
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Streamlit sayfası ayarları
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("AI Chatbot with Document Knowledge")

# Gerekli API anahtarlarını ayarla
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

def get_vector_store():

    LOGGER = get_logger(__name__)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "groove"
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings, namespace="rochester")
    return vectorstore

# Session state'i başlat
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chain' not in st.session_state:
    st.session_state.chain = None

def initialize_chain():
    # Pinecone'u başlat
    load_dotenv()
    
    vectorstore = get_vector_store()
    
    # Mevcut index'e bağlan
    #index = pinecone.Index(INDEX_NAME)
    
    # Embeddings ve ChatModel oluştur
    #embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )
    
    # Memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # ConversationalRetrievalChain oluştur
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(embedding_function=embeddings.embed_query),
        memory=memory,
        verbose=True
    )
    
    return chain

# API anahtarları girildiğinde chain'i başlat
if OPENAI_API_KEY and PINECONE_API_KEY:
    if st.session_state.chain is None:
        with st.spinner("Initializing the chatbot..."):
            st.session_state.chain = initialize_chain()
        st.success("Chatbot is ready!")

# Chat geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Ask your question..."):
    if not st.session_state.chain:
        st.error("Please enter your API keys in the sidebar first!")
    else:
        # Kullanıcı mesajını göster
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Bot yanıtını al ve göster
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": prompt})
                bot_response = response['answer']
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Sidebar'a temizle butonu ekle
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    if st.session_state.chain and hasattr(st.session_state.chain, 'memory'):
        st.session_state.chain.memory.clear()
    st.success("Chat history cleared!")
