import os
from dotenv import load_dotenv

# Import Google & Database tools
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # 1. Ask the user for a question
    print("🤖 Ask me a question about the Nebula Policy:")
    query_text = input("Question: ")

    # 2. Prepare the DB
    # (Your embedding model is working fine, so we keep this)
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # 3. Search the DB
    print("🔍 Searching documents...")
    results = db.similarity_search_with_score(query_text, k=5)

    # Combine results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 4. Ask the AI
    # UPDATED: Using "gemini-2.0-flash" which we confirmed you have access to
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("🧠 Thinking...")
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    response = model.invoke(prompt)

    # 5. Show Answer
    print("\n" + "="*50)
    print("ANSWER:")
    print(response.content)
    print("="*50)

if __name__ == "__main__":
    main()