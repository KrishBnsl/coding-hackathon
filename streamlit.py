# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from bs4 import BeautifulSoup
# import pandas as pd
# from langchain.llms import HuggingFacePipeline
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langdetect import detect
# import google.generativeai as genai
# import re
# # Configure Gemini API
# genai.configure(api_key="AIzaSyBKJoz0rbdnWkG9sKnFj32x5f5oZhA2I-o")

# # Load models
# sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# import requests
# from bs4 import BeautifulSoup
# import os

# # Directory to store documents
# os.makedirs("mental_health_docs", exist_ok=True)

# # URLs of pages to scrape
# URLS = [
#     "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
#     "https://www.who.int/news-room/fact-sheets/detail/depression",
#     "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders"
# ]

# def scrape_and_save(url, filename):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Extract all paragraphs from the article
#     paragraphs = soup.find_all("p")
#     text = "\n".join([p.get_text() for p in paragraphs])

#     # Save text to file
#     with open(f"mental_health_docs/{filename}.txt", "w", encoding="utf-8") as f:
#         f.write(text)

# # Scrape and store multiple articles
# for i, url in enumerate(URLS):
#     scrape_and_save(url, f"article_{i+1}")

# print("✅ Scraping Complete. Files saved in mental_health_docs/")

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load all scraped files
# documents = []
# doc_dir = "mental_health_docs/"

# for filename in os.listdir(doc_dir):
#     with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
#         text = f.read()
#         documents.append(text)

# # Split documents into smaller chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_text("\n".join(documents))

# # Save processed chunks
# with open("mental_health_docs/processed_texts.txt", "w", encoding="utf-8") as f:
#     f.write("\n\n".join(chunks))

# print(f"✅ Processed {len(chunks)} text chunks for retrieval.")

# # Load processed text
# with open("mental_health_docs/processed_texts.txt", "r", encoding="utf-8") as f:
#     chunks = f.readlines()

# # Convert text into vector embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Create FAISS vector store
# vector_db = FAISS.from_texts(chunks, embedding_model)
# vector_db.save_local("mental_health_vector_db")

# print("✅ Vector database created and saved.")

# vector_db = FAISS.load_local("mental_health_vector_db", embedding_model, allow_dangerous_deserialization=True)

# # Streamlit UI setup
# st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")
# st.title("🧘 Mental Health Chatbot")
# st.write("A chatbot that helps analyze your emotions, provides insights, and suggests resources.")

# # User input
# user_input = st.text_area("📝 How are you feeling today?", "", height=150)

# def query_gemini(prompt, user_language):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     lang_prompt = f"Respond in {user_language}. " + prompt
#     response = model.generate_content(lang_prompt)
#     return response.text if hasattr(response, "text") else "I'm here to listen and support you. 💙"

# def is_allowed_query(user_input):
#     critical_patterns = [r"suicide", r"harm (myself|me)", r"end my life", r"kill myself", r"severe depression"]
#     if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in critical_patterns):
#         return False, "I'm really sorry you're feeling this way. Please reach out to a professional. 💙"
#     return True, ""

# def save_to_chat_db(user_input, bot_response):
#     doc = [user_input + "\n" + bot_response]
#     vector_db.add_texts(doc)

# if st.button("Analyze & Chat"):  
#     if user_input.strip():
#         allowed, message = is_allowed_query(user_input)
#         if not allowed:
#             st.error(message)
#         else:
#             sentiment_result = sentiment_pipeline(user_input)
#             sentiment_label = sentiment_result[0]['label']
#             sentiment_score = sentiment_result[0]['score']
            
#             labels = ["happy", "sad", "angry", "anxious", "neutral"]
#             emotion_result = classifier(user_input, labels)
#             top_emotion = emotion_result['labels'][0]
#             emotion_score = emotion_result['scores'][0]
            
#             retrieved_docs = vector_db.similarity_search(user_input, k=3)
#             retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
#             user_language = detect(user_input)
#             prompt = f"Use this information to provide helpful responses:\n{retrieved_text}\n\nQ: {user_input}\nA:"
#             bot_response = query_gemini(prompt, user_language)
#             save_to_chat_db(user_input, bot_response)
            
#             # st.subheader("📊 Analysis Results")
#             # st.write(f"**Sentiment:** {sentiment_label} (Confidence: {sentiment_score:.2f})")
#             # st.write(f"**Primary Emotion:** {top_emotion} (Confidence: {emotion_score:.2f})")
#             st.subheader("💬 Chatbot Response")
#             st.write(bot_response)
#     else:
#         st.warning("Please enter some text before analyzing.")

# # Generate pseudo-accurate data for visualization
# labels = ["happy", "sad", "angry", "anxious", "neutral"]
# sentiment_counts = [random.randint(10, 50) for _ in labels]
# time_series = np.cumsum(np.random.randn(30) * 5 + 50)
# confidence_levels = np.random.uniform(0.5, 1.0, 30)


# st.markdown(
#     """
#     <style>
#     .stTextArea textarea { font-size: 16px; border-radius: 10px; padding: 10px; }
#     .stButton button { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 10px; padding: 10px; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Future Prospects
# # - 🧠 **Memory-Enabled Conversations:** Store and recall user history for personalized support.
# # - 🎭 **AI Emotion Mimicry:** Generate empathetic responses matching the user's mood.
# # - 📱 **Mobile App Integration:** Bring chatbot support to iOS & Android.
# # - 🎙️ **Voice-to-Text & Speech Support:** Enable voice-based interactions.
# # - 🌍 **Multimodal Analysis:** Combine text, voice, and facial emotion analysis.
# # - 📊 **Real-Time Mental Health Trends:** Visualize anonymized sentiment trends.
# # - 🔗 **Therapist & Crisis Helpline Integration:** Provide instant professional support links.
# # - 🏆 **Gamification for Mood Improvement:** Track mental well-being and encourage self-care.



# import streamlit as st
# import random
# import numpy as np
# import re
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from langchain.llms import HuggingFacePipeline
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langdetect import detect
# import google.generativeai as genai

# # Configure Gemini API
# genai.configure(api_key="AIzaSyBKJoz0rbdnWkG9sKnFj32x5f5oZhA2I-o")

# # Load models
# sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Load vector database
# vector_db = FAISS.load_local("mental_health_vector_db", embedding_model, allow_dangerous_deserialization=True)

# # Streamlit UI Setup
# st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="wide")
# st.markdown("<h1 style='text-align: center;'>🧘 Mental Health Chatbot</h1>", unsafe_allow_html=True)

# # Initialize chat history if not exists
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Function to query Gemini API
# def query_gemini(prompt, user_language):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     lang_prompt = prompt
#     response = model.generate_content(lang_prompt)
#     return response.text if hasattr(response, "text") else "I'm here to listen and support you. 💙"

# # Function to check critical queries
# def is_allowed_query(user_input):
#     critical_patterns = [r"suicide", r"harm (myself|me)", r"end my life", r"kill myself", r"severe depression"]
#     if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in critical_patterns):
#         return False, "I'm really sorry you're feeling this way. Please reach out to a professional. 💙"
#     return True, ""

# # User input
# user_input = st.chat_input("Type a message...")

# if user_input:
#     # Display user message
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)
    
#     # Check query
#     allowed, message = is_allowed_query(user_input)
#     if not allowed:
#         bot_response = message
#     else:
#         sentiment_result = sentiment_pipeline(user_input)
#         sentiment_label = sentiment_result[0]['label']
        
#         labels = ["happy", "sad", "angry", "anxious", "neutral"]
#         emotion_result = classifier(user_input, labels)
#         top_emotion = emotion_result['labels'][0]
        
#         retrieved_docs = vector_db.similarity_search(user_input, k=3)
#         retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
#         user_language = detect(user_input)
#         prompt = f"you are a mental health expert and are tasked with efficiently and properly helping the user to recover their mental health. Use this information to provide helpful responses:\n{retrieved_text}\n\nQ: {user_input}\nA:"
#         bot_response = query_gemini(prompt, user_language)
    
#     # Display bot response
#     with st.chat_message("assistant"):
#         st.markdown(bot_response)
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})


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

# Configure Gemini API
genai.configure(api_key="AIzaSyBKJoz0rbdnWkG9sKnFj32x5f5oZhA2I-o")  # Replace with actual API key

# Load embeddings & FAISS database
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("mental_health_vector_db", embedding_model, allow_dangerous_deserialization=True)
chat_db = FAISS.from_documents([Document(page_content="This is an empty chat history entry.", metadata={"type": "chat"})], embedding_model)

# Function to check allowed queries
def is_allowed_query(user_input):
    critical_patterns = [r"suicide", r"harm (myself|me)", r"end my life", r"kill myself", r"severe depression"]
    technical_patterns = [r"\b(machine learning|AI|programming|technology|code)\b"]

    if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in critical_patterns):
        return False, "I'm really sorry you're feeling this way. Please consider reaching out to a mental health professional. 💙"
    if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in technical_patterns):
        return False, "I'm here for mental health discussions. For technical topics, I recommend other resources. 💡"
    return True, ""

# Function to get Gemini response
def query_gemini(prompt, user_language):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Respond in english. {prompt}")
    return response.text if hasattr(response, "text") else "I'm here to support you. 💙"

# Function to extract relevant info
def extract_relevant_info(user_input, retrieved_docs):
    input_embedding = embedding_model.embed_query(user_input)
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in retrieved_docs]
    scores = cosine_similarity([input_embedding], doc_embeddings)[0]
    return " ".join([retrieved_docs[i].page_content for i in range(len(scores)) if scores[i] > 0.5]) if max(scores) > 0.5 else user_input

# Function to save chat history
def save_to_chat_db(user_input, bot_response):
    retrieved_docs = vector_db.similarity_search(user_input, k=3)
    refined_input = extract_relevant_info(user_input, retrieved_docs)
    if refined_input.strip():
        chat_db.add_documents([Document(page_content=f"User: {refined_input}\nBot: {bot_response}", metadata={"type": "chat"})])
        if len(chat_db.similarity_search("", k=200)) > 100:
            chat_db.delete([chat_db.similarity_search("", k=200)[i].id for i in range(len(chat_db.similarity_search("", k=200)) - 100)])

# Streamlit UI
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💙", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.markdown("*A supportive chatbot for mental well-being.*")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role, text = message
    with st.chat_message(role):
        st.markdown(text)

# User input
user_input = st.chat_input("How are you feeling today?")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Detect language
    try:
        user_language = detect(user_input)
    except:
        user_language = "English"

    # Check if query is allowed
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

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append(("assistant", bot_response))
    save_to_chat_db(user_input, bot_response)
