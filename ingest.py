import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Import tools

def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found. Did you make the .env file?")
        sys.exit(1)

    pdf_path = Path("Nebula_Sales_Policy_v1.pdf")
    if not pdf_path.is_file():
        print(f"❌ Error: PDF not found at {pdf_path.resolve()}")
        sys.exit(1)

    print("🚀 Starting... Reading PDF...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    print("✂️  Cutting text into pieces...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("💾 Saving data to local database...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="./chroma_db"
    )

    # ensure data is flushed to disk
    try:
        vectorstore.persist()
    except Exception:
        # some Chroma wrappers persist on creation; ignore if not available
        pass

    print("✅ Success! Your 'Brain' is ready in the 'chroma_db' folder.")

if __name__ == "__main__":
    main()