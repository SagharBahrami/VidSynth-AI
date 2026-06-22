import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------
# RAG SETTINGS
# -------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 3

# -------------------------
# CHROMADB SETTINGS
# -------------------------

APP_TITLE = "YouTube RAG Chatbot"
APP_DESCRIPTION = "Chat with any YouTube video using RAG"
GEMINI_MODEL = "gemini-3.1-pro-preview"
EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_DB_PATH = "data/chroma_db"

# -------------------------
# PODCAST (TEXT-TO-SPEECH) SETTINGS
# -------------------------

# Gemini multi-speaker TTS model.
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Maps each speaker name used in the podcast script to a Gemini prebuilt voice.
# The speaker names MUST match the names used in MULTISPEAKER_PODCAST_PROMPT.
PODCAST_SPEAKER_VOICES = {
    "Alex": "Puck",   # host
    "Maya": "Kore",   # expert guest
}