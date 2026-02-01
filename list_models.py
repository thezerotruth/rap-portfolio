import os
import sys
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not set in environment.")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def main():
    print("Authorized Models for your Key:")
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                print(f"-> {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()