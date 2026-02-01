import streamlit as st

def apply_custom_theme():
    st.markdown("""
        <style>
        /* 1. Main Background & Text */
        .stApp {
            background-color: #0E1117; /* Deep charcoal */
            color: #FFFFFF;
        }

        /* 2. Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }

        /* 3. Card/Bento Grid Style for Sections */
        div.stBlock {
            background-color: #1C2128;
            border: 1px solid #30363D;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s ease;
        }
        div.stBlock:hover {
            border-color: #10B981; /* Teal accent on hover */
            transform: translateY(-2px);
        }

        /* 4. Electric Blue/Teal Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        }

        /* 5. Headers & Inputs */
        h1, h2, h3 {
            color: #3B82F6; /* Electric blue headers */
        }
        .stTextInput>div>div>input {
            background-color: #0D1117;
            border: 1px solid #30363D;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

apply_custom_theme()
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