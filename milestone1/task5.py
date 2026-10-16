
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from sarvamai import SarvamAI

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

AUDIO_EXTENSIONS = {
    "wav", "mp3", "m4a", "mp4", "aac",
    "flac", "ogg", "webm"
}

DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "ppt", "pptx"}


def get_extension(filename):
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def read_document_using_langchain(file_path, extension):
    """Read supported documents using LangChain loaders only."""

    if extension == "pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == "docx":
        loader = Docx2txtLoader(str(file_path))

    elif extension == "txt":
        loader = TextLoader(
            str(file_path),
            autodetect_encoding=True
        )

    elif extension in {"ppt", "pptx"}:
        loader = UnstructuredPowerPointLoader(
            str(file_path),
            mode="single"
        )

    else:
        raise ValueError("Unsupported document format.")

    documents = loader.load()

    extracted_text = "\n\n".join(
        document.page_content for document in documents
    )

    return extracted_text.strip()


def convert_audio_to_text(file_path):
    """Transcribe audio using Sarvam AI."""

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise ValueError(
            "SARVAM_API_KEY is missing from milestone1/.env"
        )

    client = SarvamAI(api_subscription_key=api_key)

    with open(file_path, "rb") as audio_file:
        result = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v4",
            mode="transcribe",
            language_code="unknown",
        )

    return (result.transcript or "").strip()


@app.route("/", methods=["GET", "POST"])
def home():
    message = ""
    transcript = ""
    saved_file = ""

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if not uploaded_file or not uploaded_file.filename:
            message = "Please choose an audio or document file."

        else:
            original_name = secure_filename(uploaded_file.filename)
            extension = get_extension(original_name)

            if extension not in AUDIO_EXTENSIONS | DOCUMENT_EXTENSIONS:
                message = (
                    "Unsupported format. Upload audio, PDF, DOCX, "
                    "TXT, PPT, or PPTX."
                )

            else:
                temp_path = (
                    BASE_DIR / f"temp_{uuid4().hex}_{original_name}"
                )

                try:
                    uploaded_file.save(temp_path)

                    if extension in AUDIO_EXTENSIONS:
                        transcript = convert_audio_to_text(temp_path)
                    else:
                        transcript = read_document_using_langchain(
                            temp_path, extension
                        )

                    if not transcript:
                        raise ValueError(
                            "No text was extracted from this file."
                        )

                    output_name = (
                        f"{Path(original_name).stem}_text.txt"
                    )
                    output_path = OUTPUT_DIR / output_name

                    # Avoid overwriting an earlier output file.
                    if output_path.exists():
                        output_path = OUTPUT_DIR / (
                            f"{Path(original_name).stem}_"
                            f"{uuid4().hex[:6]}_text.txt"
                        )

                    output_path.write_text(
                        transcript,
                        encoding="utf-8"
                    )

                    saved_file = output_path.name
                    message = (
                        "Success! Extracted text saved to "
                        f"output/{saved_file}"
                    )

                except Exception as error:
                    app.logger.exception("Task 5 processing failed")
                    message = f"Processing failed: {error}"

                finally:
                    if temp_path.exists():
                        temp_path.unlink()

    return render_template(
        "task5.html",
        message=message,
        transcript=transcript,
        saved_file=saved_file,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8001,
        debug=True
    )
