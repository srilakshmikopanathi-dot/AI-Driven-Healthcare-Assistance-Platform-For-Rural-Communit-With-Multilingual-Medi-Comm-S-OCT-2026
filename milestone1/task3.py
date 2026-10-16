
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API keys
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Use image-capable models available to your API keys
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-3.1-flash-lite"

IMAGE_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ask_ai(image_path, prompt):
    image_data = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    extensions = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = extensions.get(image_path.suffix.lower())

    if not mime_type:
        raise ValueError("Unsupported image format")

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        }
    ])

    errors = []

    if OPENAI_KEY:
        try:
            model = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_KEY,
                temperature=0
            )
            return model.invoke([message]).content
        except Exception as error:
            errors.append(f"OpenAI: {error}")

    if GEMINI_KEY:
        try:
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_KEY,
                temperature=0
            )
            return model.invoke([message]).content
        except Exception as error:
            errors.append(f"Gemini: {error}")

    raise RuntimeError(
        "Both models failed or no API key was loaded. "
        + " | ".join(errors)
    )


def main():
    if not IMAGE_DIR.exists():
        print("Images folder not found:", IMAGE_DIR)
        return

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    if not images:
        print("No prescription images found.")
        return

    for image_path in images:
        print("Processing:", image_path.name)

        try:
            # Step 1: Extract and format prescription text
            extraction_prompt = """
            Read this prescription image and extract all visible text.
            Format it using these headings:

            PRESCRIPTION DETAILS
            Doctor Details:
            Patient Details:
            Medicine Details:
            Other Details:

            Preserve medicine names and dosages as written.
            Mark unreadable text as [Unclear].
            Do not guess missing information.
            This is text extraction, not medical advice.
            """

            extracted_text = ask_ai(image_path, extraction_prompt)

            # Save formatted prescription
            formatted_file = OUTPUT_DIR / (
                image_path.stem + "_formatted.txt"
            )
            formatted_file.write_text(
                str(extracted_text), encoding="utf-8"
            )

            # Step 2: Translate into Telugu and Hindi
            for language in ["Telugu", "Hindi"]:
                translation_prompt = f"""
                Translate the following extracted prescription text
                into {language}.
                Keep medicine names, numbers, strengths, and dosages
                accurate. Do not add medical advice or invent details.
                Preserve the headings and mark unclear text clearly.

                TEXT:
                {extracted_text}
                """

                # Send translation request as text
                translated_text = ask_text_ai(translation_prompt)

                output_file = OUTPUT_DIR / (
                    image_path.stem + f"_{language.lower()}.txt"
                )
                output_file.write_text(
                    str(translated_text), encoding="utf-8"
                )

                print(f"{language} saved:", output_file)

            print("Formatted text saved:", formatted_file)

        except Exception as error:
            print(f"Failed on {image_path.name}: {error}")


def ask_text_ai(prompt):
    errors = []

    if OPENAI_KEY:
        try:
            model = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_KEY,
                temperature=0
            )
            return model.invoke(prompt).content
        except Exception as error:
            errors.append(f"OpenAI: {error}")

    if GEMINI_KEY:
        try:
            model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_KEY,
                temperature=0
            )
            return model.invoke(prompt).content
        except Exception as error:
            errors.append(f"Gemini: {error}")

    raise RuntimeError("Translation failed. " + " | ".join(errors))


if __name__ == "__main__":
    main()
