
import os
import base64
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API keys from the project root .env file
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).resolve().parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"

# Images and output folders
IMAGE_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_message(image_path):
    # Convert image into Base64
    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(image_path.suffix.lower())

    if mime_type is None:
        raise ValueError("Unsupported image format")

    prompt = """
    Extract the visible text from this prescription image.
    Organize the text into clear sections.
    Preserve medicine names and dosage text as written.
    Mark unreadable text as [Unclear].
    Do not guess missing information.
    This is text extraction, not medical advice.
    """

    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                },
            },
        ]
    )


def read_prescription(image_path):
    message = build_message(image_path)

    # Try OpenAI using LangChain
    if OPENAI_API_KEY:
        try:
            model = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )

            response = model.invoke([message])
            return f"OpenAI ({OPENAI_MODEL})", response.content

        except Exception as error:
            print("OpenAI failed:", error)
            print("Trying Gemini...")

    # Fallback to Gemini using LangChain
    if GEMINI_API_KEY:
        try:
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0,
            )

            response = model.invoke([message])
            return f"Gemini ({GEMINI_MODEL})", response.content

        except Exception as error:
            print("Gemini failed:", error)

    raise RuntimeError(
        "No model could process the image. Check API keys and model access."
    )


def main():
    print("=" * 50)
    print("AI PRESCRIPTION TEXT EXTRACTION")
    print("=" * 50)

    if not IMAGE_DIR.exists():
        print("Images folder not found:", IMAGE_DIR)
        return

    image_files = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if len(image_files) < 2:
        print("Please add two prescription images to the images folder.")
        return

    for image_path in image_files[:2]:
        print(f"\nProcessing: {image_path.name}")

        try:
            model_name, extracted_text = read_prescription(image_path)

            output_path = OUTPUT_DIR / f"{image_path.stem}.txt"

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(f"Image: {image_path.name}\n")
                file.write(f"Model: {model_name}\n")
                file.write("=" * 50 + "\n\n")
                file.write(str(extracted_text))

            print("Model used:", model_name)
            print("Output saved:", output_path)
            print("Extracted text:")
            print(extracted_text)

        except Exception as error:
            print(f"Failed to process {image_path.name}: {error}")


if __name__ == "__main__":
    main()

