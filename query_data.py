import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Import Google & Database tools

load_dotenv()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    try:
        # Get question from CLI arg or prompt
        if len(sys.argv) > 1:
            query_text = " ".join(sys.argv[1:])
        else:
            print("🤖 Ask me a question about the Nebula Policy:")
            query_text = input("Question: ").strip()
        if not query_text:
            print("No question provided. Exiting.")
            return

        # Prepare DB
        embedding_model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
        chroma_dir = os.getenv("CHROMA_DIR", "./chroma_db")
        embedding_function = GoogleGenerativeAIEmbeddings(model=embedding_model)
        db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)

        # Search the DB
        print("🔍 Searching documents...")
        results = db.similarity_search_with_score(query_text, k=5)
        if not results:
            print("No relevant documents found.")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        # Build prompt and call model
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        print("🧠 Thinking...")
        model_name = os.getenv("LLM_MODEL", "gemini-pro")
        model = ChatGoogleGenerativeAI(model=model_name)
        response = model.invoke(prompt)

        # Show answer
        print("\n" + "=" * 50)
        print("ANSWER:")
        # response may be an object with .content or a string
        answer = getattr(response, "content", response)
        print(answer)
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()