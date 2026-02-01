import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import the logic we used yesterday

# 1. Load the "Brain"
load_dotenv()

# Setup Page
st.set_page_config(page_title="Nebula AI Deal Desk", page_icon="🤖")
st.title("🤖 Nebula AI Deal Desk Assistant")
st.write("Ask questions about the Global Sales Policy without reading the PDF!")

# 2. Connect to Database (Cached so it doesn't reload every click)
@st.cache_resource
def load_db():
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Make sure this points to your existing 'chroma_db' folder
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db

db = load_db()

# 3. Handle the User Question
query_text = st.text_input("Enter your question:", placeholder="e.g., Who approves a 20% discount?")

if query_text:
    # A. Search DB
    st.write("🔍 Searching policy documents...")
    results = db.similarity_search_with_score(query_text, k=3)
    
    # Context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # B. Generate Answer
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    ---
    Answer the question based on the above context: {question}
    """
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # C. Call Google AI
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    response = model.invoke(prompt)
    
    # D. Display Answer
    st.success("Answer Found:")
    st.write(response.content)
    
    # Optional: Show what text it read (for debugging)
    with st.expander("See referenced policy text"):
        st.write(context_text)