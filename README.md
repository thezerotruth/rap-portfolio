# 🤖 Nebula AI Deal Desk Assistant

A RAG (Retrieval-Augmented Generation) application that acts as an automated "Deal Desk" for sales teams. It ingests internal sales policy PDF documents and uses Google Gemini to answer questions about discounts, SLAs, and approval hierarchies.

## 🛠️ Tech Stack
*   **Python 3.12**
*   **LangChain:** For document processing and retrieval logic.
*   **Google Gemini (2.0 Flash):** LLM for reasoning and answer generation.
*   **ChromaDB:** Vector database for semantic search.
*   **Streamlit:** Front-end user interface.

## 🚀 How it Works
1.  **Ingestion:** The app loads `Nebula_Sales_Policy_v1.pdf`, splits it into chunks, and stores vector embeddings in ChromaDB.
2.  **Retrieval:** When a user asks a question, the system performs a similarity search to find the relevant policy sections.
3.  **Generation:** The relevant context + user question are sent to Gemini 2.0 to generate a natural language response.

## 📂 Project Structure
*   `ingest.py`: Script to process the PDF and build the vector database.
*   `app.py`: The Streamlit web application.
*   `requirements.txt`: List of dependencies.

## 💻 How to Run Locally
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Add your API Key to a `.env` file: `GOOGLE_API_KEY=your_key`
4.  Build the database: `python ingest.py`
5.  Run the app: `streamlit run app.py`
