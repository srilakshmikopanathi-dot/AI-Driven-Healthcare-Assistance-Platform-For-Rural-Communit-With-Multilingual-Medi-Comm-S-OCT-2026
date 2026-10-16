
import os
import base64
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load the API key from milestone1/.env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)

SUPPORTED_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "te-IN": "Telugu",
}


def detect_language(text):
    """Basic detection for Telugu, Hindi, and English."""
    if any("\u0c00" <= char <= "\u0c7f" for char in text):
        return "te-IN"

    if any("\u0900" <= char <= "\u097f" for char in text):
        return "hi-IN"

    return "en-IN"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/speak", methods=["POST"])
def speak():
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        return jsonify({
            "error": "SARVAM_API_KEY is missing from milestone1/.env"
        }), 500

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    language = data.get("language", "auto")

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    if len(text) > 2500:
        return jsonify({
            "error": "Text must be 2500 characters or fewer."
        }), 400

    if language == "auto":
        language_code = detect_language(text)
    elif language in SUPPORTED_LANGUAGES:
        language_code = language
    else:
        return jsonify({
            "error": "Please select a supported language."
        }), 400

    try:
        # Connect to Sarvam AI
        client = SarvamAI(api_subscription_key=api_key)

        # Convert the entered text into speech
        result = client.text_to_speech.convert(
            text=text,
            language_code=language_code,
            model="bulbul:v3",
            speaker="shubh",
            output_audio_codec="wav",
        )

        # Decode the audio returned by Sarvam AI
        audio_base64 = "".join(result.audios)
        audio_bytes = base64.b64decode(audio_base64)

        # Return audio to the browser
        return Response(
            audio_bytes,
            mimetype="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as error:
        app.logger.exception("Sarvam AI text-to-speech failed")

        return jsonify({
            "error": "Speech generation failed.",
            "details": str(error),
        }), 502


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )
