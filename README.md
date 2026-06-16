# 🎬 VidSynth AI

VidSynth AI is a Streamlit application that transforms YouTube videos into AI-generated study notes, summaries, key topics, and later an interactive video chatbot using RAG.

## 🚀 Features

### Current Features

- Accepts a YouTube video URL
- Fetches the video transcript
- Generates study notes from the transcript
- Creates a short summary
- Extracts key topics and key notes
- Displays results in a clean Streamlit interface

### Planned Features

- Chat with a YouTube video using RAG
- Store transcript chunks in ChromaDB
- Retrieve relevant chunks based on user questions
- Generate answers using Gemini
- Add podcast/audio generation
- Improve transcript language support

## 🧠 What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

In this project, RAG means:

1. Get the YouTube transcript
2. Split the transcript into chunks
3. Convert chunks into embeddings
4. Store chunks in ChromaDB
5. Retrieve the most relevant chunks for a user question
6. Send those chunks to Gemini
7. Generate an answer based on the video content

## 🛠️ Tech Stack

- Python
- Streamlit
- Gemini API
- LangChain
- ChromaDB
- YouTube Transcript API
- python-dotenv

## 📁 Project Structure

```text
project3/
│
├── app.py
├── services.py
├── config.py
├── prompts.py
├── requirements.txt
├── .env
├── README.md
│
└── data/
    └── chroma_db/