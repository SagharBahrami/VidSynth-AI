# 🎬 VidSynth AI

VidSynth AI is an AI-powered Streamlit application that transforms YouTube videos into study notes, searchable video chat, and audio-assisted learning.

Users can paste a YouTube URL, generate organized notes, chat with the video using RAG with hybrid retrieval, ask questions by voice, generate a multi-speaker podcast, and listen to audio versions of generated content.

---

## ✨ Features

### 📝 Notes For You

Generate structured study notes from a YouTube video transcript.

The notes include:

- Short Summary
- Key Topics
- Content Outline
- Detailed Study Notes
- Important Takeaways

Users can choose the language of the generated notes by entering a language code such as:

```text
en
fa
es
fr
ar
```

---

### 💬 Chat with Video

Chat with a YouTube video after processing its transcript.

The app uses RAG, which stands for:

```text
Retrieval-Augmented Generation
```

The chat flow is:

```text
YouTube transcript
→ Translate transcript if needed
→ Split transcript into chunks
→ Store chunks in ChromaDB (dense) and build a BM25 index (sparse)
→ User asks a question
→ Hybrid retrieval: dense + BM25 candidates fused with RRF
→ Gemini generates an answer from the top chunks
```

This helps the chatbot answer based on the video transcript instead of giving unrelated general answers.

Retrieval is **hybrid**: it combines semantic (dense) search with keyword (BM25) search and merges the two rankings using Reciprocal Rank Fusion (RRF). See the [Hybrid Retrieval](#-hybrid-retrieval-dense--bm25--rrf) section for details.

---

### 🎙️ Voice Questions

Users can ask questions using their microphone.

The app records the user's voice, converts the audio into text, and sends the transcribed question to the same RAG pipeline used for typed questions.

Flow:

```text
Voice question
→ Speech-to-text
→ Retrieve relevant transcript chunks
→ Generate answer
```

---

### 🔊 Audio Generation

VidSynth AI can generate audio from AI-generated text.

Current audio support includes:

- Audio version of study notes
- Audio version of chatbot answers

The app uses `gTTS` for text-to-speech.

Generated audio files are saved locally in:

```text
data/audio/
```

If a selected language is not supported by the audio provider, the app can generate English audio instead.

---

### 🎙️ Podcast (Multi-Speaker)

VidSynth AI can turn a YouTube video into a short two-speaker podcast episode.

The app generates a natural conversation between two speakers:

```text
Alex — the host
Maya — the expert guest
```

The podcast flow is:

```text
YouTube transcript
→ Translate transcript if needed
→ Generate study notes with Gemini
→ Write a two-speaker podcast script with Gemini
→ Synthesize multi-speaker audio with Gemini text-to-speech
```

Unlike the `gTTS` audio (which produces a single MP3 voice), the podcast uses
Google Gemini's multi-speaker text-to-speech to give each speaker a distinct
voice. The generated podcast is saved as a `.wav` file in `data/audio/`, and the
script is also viewable in the app.

> The podcast feature uses Gemini's paid text-to-speech model, so a billing-enabled
> Google API key is required to generate podcast audio.

---

### 🌍 Multilingual Support

VidSynth AI separates two different language concepts:

```text
Transcript language = the language available from YouTube captions
Note / response language = the language selected by the user
```

The app can fetch the available transcript, translate it into the selected response language, and then generate notes or build a chat knowledge base.

---

## 🧠 How It Works

### Notes Generation Flow

```text
YouTube URL
→ Extract video ID
→ Detect transcript language
→ Fetch transcript
→ Translate transcript if needed
→ Generate structured notes with Gemini
→ Optionally generate audio
```

### Chat with Video Flow

```text
YouTube URL
→ Extract video ID
→ Detect transcript language
→ Fetch transcript
→ Translate transcript if needed
→ Split transcript into chunks
→ Store chunks in ChromaDB (dense) and build a BM25 index (sparse)
→ User asks a typed or voice question
→ Hybrid retrieval (dense + BM25) fused with RRF, keep the top FINAL_K chunks
→ Generate answer with Gemini
→ Optionally generate audio answer
```

### Podcast Generation Flow

```text
YouTube URL
→ Extract video ID
→ Detect transcript language
→ Fetch transcript
→ Translate transcript if needed
→ Generate study notes with Gemini
→ Write a two-speaker podcast script with Gemini
→ Synthesize multi-speaker audio with Gemini text-to-speech
→ Play and download the podcast
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini (text + multi-speaker text-to-speech)
- ChromaDB (dense vector search)
- rank_bm25 (sparse keyword search for hybrid retrieval)
- YouTube Transcript API
- gTTS
- SpeechRecognition
- python-dotenv

---

## 📁 Project Structure

```text
VidSynth-AI/
├── app.py
├── services.py
├── prompts.py
├── config.py
├── requirements.txt
├── README.md
├── .env
└── data/
    ├── chroma_db/
    └── audio/
```

### File Responsibilities

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and app workflow |
| `services.py` | Transcript extraction, translation, hybrid retrieval (dense + BM25 + RRF), RAG, audio generation, podcast generation, and speech-to-text functions |
| `prompts.py` | Prompt templates for notes, translation, RAG answers, and the multi-speaker podcast script |
| `config.py` | API keys, model names (including the Gemini TTS model and podcast speaker voices), chunk settings, hybrid retrieval settings, and ChromaDB path |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/SagharBahrami/VidSynth-AI.git
cd VidSynth-AI
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory.

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The key must belong to a Google project with **billing enabled** to use the
podcast (multi-speaker text-to-speech) feature.

Do not push your `.env` file to GitHub.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
data/chroma_db/
data/audio/
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser.

Usually it will be:

```text
http://localhost:8501
```

---

## 🧪 How to Use

### Generate Notes

1. Paste a YouTube URL.
2. Enter the desired note language code.
3. Select `Notes For You`.
4. Click `✨ Start Processing`.
5. Read the generated notes.
6. Optionally generate an audio version.

---

### Chat with a Video

1. Paste a YouTube URL.
2. Enter the desired response language code.
3. Select `Chat with Video`.
4. Click `✨ Start Processing`.
5. Ask questions using text or voice.
6. Read or listen to the AI response.

---

### Generate a Podcast

1. Paste a YouTube URL.
2. Enter the desired podcast language code.
3. Select `Podcast`.
4. Click `✨ Start Processing`.
5. Listen to the generated two-speaker podcast.
6. Download the `.wav` file or expand the script to read it.

---

## 🔎 RAG Explanation

RAG stands for:

```text
Retrieval-Augmented Generation
```

In VidSynth AI, RAG works like this:

### 1. Retrieval

The app uses **hybrid retrieval** to find transcript chunks related to the user's question
(see the [Hybrid Retrieval](#-hybrid-retrieval-dense--bm25--rrf) section below).

### 2. Augmented

The retrieved transcript chunks are added to the prompt.

### 3. Generation

Gemini generates an answer using the retrieved transcript context.

This allows the chatbot to answer questions based on the actual video content.

---

## 🔀 Hybrid Retrieval (Dense + BM25 + RRF)

Instead of relying on a single search method, VidSynth AI combines two complementary ones:

```text
Dense  (ChromaDB vector search) → finds chunks by MEANING   (good for paraphrased questions)
Sparse (BM25 keyword search)    → finds chunks by EXACT WORDS (good for names, terms, codes)
```

Each retriever proposes a set of candidate chunks. The two rankings are then merged with
**Reciprocal Rank Fusion (RRF)**, which scores each chunk by its *rank* in each list
(`1 / (RRF_K + rank)`) rather than by raw scores. This avoids the problem that a vector
distance and a BM25 score are on completely different scales. A chunk ranked highly by
**both** retrievers rises to the top.

```text
question
  → dense search → CANDIDATE_K chunks  ┐
  → BM25 search  → CANDIDATE_K chunks   ├─► RRF fuse & re-rank ─► keep top FINAL_K ─► Gemini
                                        ┘
```

The relevant settings live in `config.py`:

| Setting | Meaning |
|---|---|
| `CANDIDATE_K` | How many chunks **each** retriever proposes before fusion (wider = better quality, ~free) |
| `FINAL_K` | How many fused chunks are sent to the LLM (lower = fewer tokens) |
| `RRF_K` | RRF constant (default `60`, from the original RRF paper) |

Because BM25 runs locally, hybrid retrieval adds no token cost. Token usage is controlled by
`FINAL_K`: better retrieval means a small `FINAL_K` (e.g. `2`) usually answers as well as a
larger dense-only `TOP_K` did, so fewer chunks reach the LLM.

---

## 🎧 Audio Features

VidSynth AI uses `gTTS` to generate MP3 audio from text.

Audio can be generated for:

- Study notes
- Chatbot answers

Generated audio files are saved in:

```text
data/audio/
```

Some languages may not be supported by `gTTS`. In that case, the app can generate English audio instead.

---

## 🎙️ Voice Question Feature

VidSynth AI uses `SpeechRecognition` to convert recorded voice questions into text.

The voice question flow is:

```text
Record audio
→ Convert speech to text
→ Send text question to RAG
→ Generate AI answer
```

For best results, the spoken question language should match the selected response language code.

---

## ⚠️ Limitations

- The YouTube video must have an available transcript or captions.
- Speech recognition accuracy depends on microphone quality, background noise, and language support.
- gTTS language support may vary by language.
- ChromaDB data is stored locally.
- Generated audio files are stored locally.
- Very long videos may take more time to translate, chunk, embed, and process.

---

## 🚀 Future Improvements

Possible future improvements:

- More than two podcast speakers
- Better audio voice options
- Better UI styling
- Source chunk display for chatbot answers
- Chat history export
- More advanced multilingual speech support
- Streamlit Cloud deployment
- User authentication
- Automatic cleanup for old generated audio files

---

## 👩‍💻 Author

Built by **Saghar Bahrami**.

GitHub: [SagharBahrami](https://github.com/SagharBahrami)

---

## 📄 License

This project is for educational and portfolio purposes.

A license file can be added later to define usage permissions more clearly.