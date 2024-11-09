import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Streamlit sayfası ayarları
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("AI Chatbot with Document Knowledge")

# Gerekli API anahtarlarını ayarla
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
PINECONE_API_KEY = st.sidebar.text_input("Pinecone API Key", type="password")
PINECONE_ENV = st.sidebar.text_input("Pinecone Environment")
INDEX_NAME = st.sidebar.text_input("Pinecone Index Name")

# Session state'i başlat
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chain' not in st.session_state:
    st.session_state.chain = None

def initialize_chain():
    # Pinecone'u başlat
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    
    # Mevcut index'e bağlan
    index = pinecone.Index(INDEX_NAME)
    
    # Embeddings ve ChatModel oluştur
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Memory oluştur
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # ConversationalRetrievalChain oluştur
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.as_retriever(embedding_function=embeddings.embed_query),
        memory=memory,
        verbose=True
    )
    
    return chain

# API anahtarları girildiğinde chain'i başlat
if OPENAI_API_KEY and PINECONE_API_KEY and PINECONE_ENV and INDEX_NAME:
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
