import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import re
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

genai.configure(api_key="AIzaSyBKJoz0rbdnWkG9sKnFj32x5f5oZhA2I-o")  
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("mental_health_vector_db", embedding_model, allow_dangerous_deserialization=True)
chat_db = FAISS.from_documents([Document(page_content="This is an empty chat history entry.", metadata={"type": "chat"})], embedding_model)

def is_allowed_query(user_input):
    critical_patterns = [r"suicide", r"harm (myself|me)", r"end my life", r"kill myself", r"severe depression"]
    technical_patterns = [r"\b(machine learning|AI|programming|technology|code)\b"]

    if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in critical_patterns):
        return False, "I'm really sorry you're feeling this way. Please consider reaching out to a mental health professional. "
    if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in technical_patterns):
        return False, "I'm here for mental health discussions. For technical topics, I recommend other resources. "
    return True, ""

def query_gemini(prompt, user_language):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Respond in english. {prompt}")
    return response.text if hasattr(response, "text") else "I'm here to support you. "

def extract_relevant_info(user_input, retrieved_docs):
    input_embedding = embedding_model.embed_query(user_input)
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in retrieved_docs]
    scores = cosine_similarity([input_embedding], doc_embeddings)[0]
    return " ".join([retrieved_docs[i].page_content for i in range(len(scores)) if scores[i] > 0.5]) if max(scores) > 0.5 else user_input

def save_to_chat_db(user_input, bot_response):
    retrieved_docs = vector_db.similarity_search(user_input, k=3)
    refined_input = extract_relevant_info(user_input, retrieved_docs)
    if refined_input.strip():
        chat_db.add_documents([Document(page_content=f"User: {refined_input}\nBot: {bot_response}", metadata={"type": "chat"})])
        if len(chat_db.similarity_search("", k=200)) > 100:
            chat_db.delete([chat_db.similarity_search("", k=200)[i].id for i in range(len(chat_db.similarity_search("", k=200)) - 100)])

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💙", layout="centered")
st.title(" Mental Health Chatbot")
st.markdown("*A supportive chatbot for mental well-being.*")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role, text = message
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("How are you feeling today?")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        user_language = detect(user_input)
    except:
        user_language = "English"

    allowed, message = is_allowed_query(user_input)
    if not allowed:
        bot_response = message
    else:
        retrieved_docs = vector_db.similarity_search(user_input, k=3)
        chat_context = "\n\n".join([doc.page_content for doc in chat_db.similarity_search(user_input, k=2)])
        system_prompt = f"Previous conversation:\n{chat_context}\n\n" if chat_context else ""
        context = f"{retrieved_docs}\n\n{chat_context}" if retrieved_docs else chat_context
        prompt = f"{system_prompt}Use the following information to provide empathetic and professional responses:\n{context}\n\nQ: {user_input}\nA:"
        bot_response = query_gemini(prompt, user_language)

    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append(("assistant", bot_response))
    save_to_chat_db(user_input, bot_response)
