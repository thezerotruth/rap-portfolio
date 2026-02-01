import os
from dotenv import load_dotenv

import google.generativeai as genai

# Load your key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Key not found!")
else:
    print(f"✅ Key found. Asking Google for available models...")
    genai.configure(api_key=api_key)
    
    try:
        # List all models that support generating text
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"AVAILABLE: {m.name}")
    except Exception as e:
        print(f"❌ Error talking to Google: {e}")