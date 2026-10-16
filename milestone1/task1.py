

import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai

# Load API keys from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

# Models used
OPENAI_MODEL = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-2.5-flash"


# 1. List OpenAI models
def list_openai_models():
    print("\n" + "=" * 60)
    print("AVAILABLE OPENAI MODELS")
    print("=" * 60)

    client = OpenAI(api_key=OPENAI_API_KEY)
    models = client.models.list()

    for model in sorted(models.data, key=lambda x: x.id):
        print(model.id)


# 2. List Gemini models
def list_gemini_models():
    print("\n" + "=" * 60)
    print("AVAILABLE GEMINI MODELS")
    print("=" * 60)

    client = genai.Client(api_key=GEMINI_API_KEY)

    for model in client.models.list():
        if model.supported_actions:
            if "generateContent" in model.supported_actions:
                print(model.name)


# 3. Ask OpenAI using LangChain
def ask_chatgpt(query):
    print("\n" + "=" * 60)
    print("OPENAI RESPONSE")
    print("=" * 60)

    model = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    response = model.invoke(query)
    print(response.content)


# 4. Ask Gemini using LangChain
def ask_gemini(query):
    print("\n" + "=" * 60)
    print("GEMINI RESPONSE")
    print("=" * 60)

    model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    response = model.invoke(query)
    print(response.content)


# 5. Main function
def main():
    query = """
    Explain how an AI healthcare assistance platform
    can help rural communities in 3 simple points.
    """

    # List available models
    list_openai_models()
    list_gemini_models()

    # Ask both models the same question
    ask_chatgpt(query)
    ask_gemini(query)


if __name__ == "__main__":
    main()