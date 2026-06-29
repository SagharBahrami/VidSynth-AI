from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from config import (GEMINI_MODEL,
                    CHUNK_OVERLAP,
                    CHUNK_SIZE,
                    TOP_K,
                    CHROMA_DB_PATH,
                    EMBEDDING_MODEL,
                    GOOGLE_API_KEY,
                    GEMINI_TTS_MODEL,
                    PODCAST_SPEAKER_VOICES,
                    CANDIDATE_K,
                    FINAL_K,
                    RRF_K
                    )

from prompts import RAG_PROMPT, NOTES_PROMPT, TRANSLATION_PROMPT, MULTISPEAKER_PODCAST_PROMPT, SPEAKER_PLAN_PROMPT

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from rank_bm25 import BM25Okapi

from gtts import gTTS
from gtts.lang import tts_langs
import speech_recognition as sr
import tempfile
import os
import wave

from google import genai
from google.genai import types

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# Raw google-genai client used for multi-speaker text-to-speech.
# (langchain does not expose Gemini's audio output, so we call the SDK directly.)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

def extract_video_id(url: str) -> str | None:
    
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    
    return None

def get_transcript_language_code(video_id: str) -> str:
    
    if not video_id:
        raise ValueError("Not a valid video ID.")
    
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.list(video_id)

    # Try English first if available
    try:
        transcript_obj = transcript_list.find_transcript(["en"])
        return transcript_obj.language_code
    
    except:
        # If English is not available, use the first available transcript
        for transcript in transcript_list:
            return transcript.language_code

    raise ValueError("No transcript found for this video.")

def normalize_language_code(language_code: str) -> str:
    return language_code.strip().lower().split("-")[0]


def get_transcript(video_id: str, transcript_language_code: str) -> str:
    
    if not video_id:
       raise ValueError("Not a valid video ID.")
    
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=[transcript_language_code])
    transcript_parts = []
    for snippet in fetched_transcript:
        transcript_parts.append(snippet.text)
    transcript = " ".join(transcript_parts)
    
    return transcript

def extract_response_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))

        return "\n\n".join(text_parts)

    return str(content)


translation_prompt = ChatPromptTemplate.from_template(TRANSLATION_PROMPT)
translation_chain = (
    translation_prompt
    |llm
    |RunnableLambda(extract_response_text)
)


def translate_transcript_to_language_code(transcript: str, transcript_language_code: str, note_language_code: str) -> str:
    
    if not transcript:
        raise ValueError("Transcript is empty.")

    if normalize_language_code(transcript_language_code) == normalize_language_code(note_language_code):
        return transcript
    
    translated_transcript = translation_chain.invoke({
        "transcript": transcript,
        "transcript_language_code": transcript_language_code,
        "note_language_code": note_language_code
    })

    return translated_transcript


def chunk_transcript(video_id: str, transcript: str, note_language_code: str) -> list[Document]:
    document = Document(
        page_content=transcript,
        metadata={
            "video_id": video_id,
            "note_language_code": note_language_code
        }
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents([document])
    
    for index, chunk in enumerate(chunks):
        
        chunk.metadata["chunk_index"] = index

    return chunks


def store_in_chromadb(chunks: list[Document], video_id: str) -> Chroma:
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=video_id
    )
    return vector_store


def build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    # BM25 is a keyword (sparse) ranking algorithm. It needs the corpus tokenized
    # into lists of words. We lowercase and split on whitespace so matching is
    # case-insensitive. The index order matches the `chunks` list order, which is
    # how we map scores back to the original Documents later.
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def bm25_search(question: str, bm25: BM25Okapi, chunks: list[Document], k: int) -> list[Document]:
    # Score every chunk against the question, then return the k highest-scoring.
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Rank chunk positions by score (highest first), then map back to Documents.
    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:k]]


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], rrf_k: int = RRF_K, top_k: int = FINAL_K) -> list[Document]:
    # RRF merges several ranked lists by RANK, not score, so we never have to make
    # a vector distance and a BM25 score comparable. Each document earns
    # 1 / (rrf_k + rank) from every list it appears in; we sum those contributions.
    # A document ranked highly by BOTH retrievers ends up on top.
    scores = {}
    docs_by_key = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content  # identify a chunk by its text
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    ordered_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ordered_keys[:top_k]]


def retrieve_relevant_chunks(question: str, vector_store: Chroma, bm25: BM25Okapi, chunks: list[Document]) -> list[Document]:
    # Dense (semantic) candidates from the vector store.
    dense_results = vector_store.similarity_search(question, k=CANDIDATE_K)

    # Sparse (keyword) candidates from BM25.
    sparse_results = bm25_search(question, bm25, chunks, k=CANDIDATE_K)

    # Fuse both rankings with RRF and keep only the top FINAL_K. Sending fewer,
    # better chunks to the LLM is what actually reduces token consumption.
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    return fused_results


notes_prompt = ChatPromptTemplate.from_template(NOTES_PROMPT)
notes_chain = (
    notes_prompt
    | llm
    | RunnableLambda(extract_response_text)
)

def generate_video_notes(transcript: str, note_language_code: str) ->str:    
    if not transcript:
        raise ValueError("Transcript is empty.")

    notes = notes_chain.invoke({
        "context": transcript,
        "note_language_code": note_language_code
    })
    return notes


rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
rag_chain = (
    rag_prompt
    | llm
    | RunnableLambda(extract_response_text)
)


def generate_rag_answer(question, relevant_chunks: list[Document], note_language_code: str) -> str:
    if not question:
        raise ValueError("Question is empty")
    
    if not relevant_chunks:
        raise ValueError("No relevant chunks found")
    context = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )
    answer = rag_chain.invoke({
        "context": context,
        "question": question,
        "note_language_code": note_language_code
    })
    
    return answer

def generate_audio_from_text(text: str, language_code: str, video_id: str, audio_type, output_folder: str = "data/audio"):
    
    if not text:
        raise ValueError("Note is empty to generate audion of it!")
    
    if not video_id:
        raise ValueError("Video ID is empty. Cannot name audio file.")
    
    note_language_code = language_code.strip().lower()
    
    supported_languages = tts_langs()
    
    audio_text = text
    audio_language_code = language_code
    
    
    if language_code not in supported_languages:
        
        audio_text = translation_chain.invoke({
            "transcript": text,
            "transcript_language_code": language_code,
            "note_language_code": "en"
        })
    
   
    
        audio_language_code = "en"

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        f"{video_id}_{audio_type}_{audio_language_code}.mp3"
    )

    tts = gTTS(
        text=audio_text,
        lang=audio_language_code
    )

    tts.save(output_path)

    return output_path, audio_language_code

def get_speech_language_code(language_code: str) -> str:
    
    language_code = language_code.strip().lower()
    
    language_map = {
        "en": "en-US",
        "fa": "fa-IR",
        "es": "es-ES",
        "fr": "fr-FR",
        "ar": "ar-SA",
        "de": "de-DE",
        "it": "it-IT",
    }

    return language_map.get(language_code, language_code)

def convert_audio_to_text(audio_file, language_code: str):
    
    if audio_file is None:
        raise ValueError("No audio file found!")
    
    recognizer = sr.Recognizer()
    
    speech_language_code = get_speech_language_code(language_code)
    
    audio_bytes = audio_file.getvalue()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        
    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(
            audio_data,
            language=speech_language_code

        )
        return text
    except sr.UnknownValueError:
        raise ValueError("Sorry, I could not understand the audio.")
    except sr.RequestError:
        raise ValueError("Speech recognition service is unavailable right now.")

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)    

multispeaker_podcast_prompt = ChatPromptTemplate.from_template(
    MULTISPEAKER_PODCAST_PROMPT
)          

multispeaker_podcast_chain = (
    multispeaker_podcast_prompt
    | llm
    | RunnableLambda(extract_response_text)
)


# Chain that asks Gemini to read the transcript and return the major topics as
# JSON. JsonOutputParser turns that JSON text into a Python dict for us, so the
# result is a real dict (not a string we have to parse by hand).
speaker_plan_prompt = ChatPromptTemplate.from_template(SPEAKER_PLAN_PROMPT)
speaker_plan_chain = (
    speaker_plan_prompt
    | llm
    | JsonOutputParser()
)


def build_speaker_section(speakers: list) -> str:
    # speakers is a list of (name, role, voice) tuples, with the host first.
    # We turn it into instruction text that gets injected into the podcast prompt.
    names = [name for name, role, voice in speakers]

    lines = [f"Use exactly {len(speakers)} speakers: {', '.join(names)}."]
    for name, role, voice in speakers:
        lines.append(f"- {name} is the {role}.")

    return "\n".join(lines)


def plan_speakers(transcript: str) -> list:
    # Ask Gemini which major topics the transcript covers. Each topic becomes a
    # specialist guest; the first roster voice is always the host.
    try:
        plan = speaker_plan_chain.invoke({"transcript": transcript})
        topics = plan["topics"]
    except Exception:
        # If the analysis call fails or returns invalid JSON, fall back to a
        # single generic guest so the podcast still works (2 speakers total).
        # Never let an unpredictable LLM response crash the whole feature.
        topics = [{"specialist_role": "expert"}]

    # Enforce the cap in code too: at most 3 specialists (+ 1 host = 4 speakers).
    topics = topics[:3]
    if not topics:
        topics = [{"specialist_role": "expert"}]

    roster = list(PODCAST_SPEAKER_VOICES.items())  # ordered [(name, voice), ...]

    # The first roster entry is always the host (no specialist topic).
    host_name, host_voice = roster[0]
    speakers = [(host_name, "host", host_voice)]

    # One guest per topic, taking the next available voice from the roster.
    for index, topic in enumerate(topics):
        guest_name, guest_voice = roster[1 + index]
        role = topic.get("specialist_role", "expert")
        speakers.append((guest_name, role, guest_voice))

    return speakers
            
def generate_multispeaker_podcast_script(notes: str, speakers: list, podcast_language_code: str = "en") -> str:
    
    if not notes:
        raise ValueError("Notes are empty. Cannot generate podcast script.")
    
    speaker_section = build_speaker_section(speakers)

    podcast_script = multispeaker_podcast_chain.invoke({
        "notes": notes,
        "podcast_language_code": podcast_language_code,
        "speaker_section": speaker_section
    })
    
    return podcast_script


# Helper to save audio
def save_wave_file(filename: str, pcm_data, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    if not filename:
        raise ValueError("Filename is required.")

    if pcm_data is None:
        raise ValueError("Audio data is required.")

    if channels <= 0:
        raise ValueError("Channels must be greater than 0.")

    if rate <= 0:
        raise ValueError("Sample rate must be greater than 0.")

    if sample_width <= 0:
        raise ValueError("Sample width must be greater than 0.")

    output_folder = os.path.dirname(filename)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    audio_bytes = bytes(pcm_data)

    with wave.open(filename, "wb") as wave_file:
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(rate)
        wave_file.writeframes(audio_bytes)


def parse_podcast_script(podcast_script: str, valid_names: set) -> list:
    # Turn the script text into a list of (speaker_name, line_text) pairs.
    # Each line looks like "Alex: some words" — split on the FIRST colon only.
    segments = []

    for raw_line in podcast_script.splitlines():
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        name, text = line.split(":", 1)
        name = name.strip()
        text = text.strip()

        # Ignore stray lines whose prefix is not one of our real speakers.
        if name in valid_names and text:
            segments.append((name, text))

    return segments


def generate_podcast_audio(podcast_script: str, speakers: list, video_id: str, output_folder: str = "data/audio") -> str:

    if not podcast_script:
        raise ValueError("Podcast script is empty. Cannot generate podcast audio.")

    if not video_id:
        raise ValueError("Video ID is empty. Cannot name podcast file.")

    if not speakers:
        raise ValueError("No speakers provided. Cannot generate podcast audio.")

    # Map each speaker name to its voice, e.g. {"Alex": "Puck", "Maya": "Kore"}.
    voice_by_name = {name: voice for name, role, voice in speakers}

    segments = parse_podcast_script(podcast_script, set(voice_by_name.keys()))
    if not segments:
        raise ValueError("Could not find any speaker lines in the podcast script.")

    # Gemini multi-speaker TTS supports only 2 voices, so to allow more speakers
    # we synthesize ONE line at a time with that speaker's voice, then stitch the
    # raw PCM chunks together. All chunks share the same 24 kHz / 16-bit format,
    # so joining the bytes plays them back to back.
    audio_chunks = []

    for name, text in segments:
        voice_name = voice_by_name[name]

        response = genai_client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                ),
            ),
        )

        pcm_chunk = response.candidates[0].content.parts[0].inline_data.data
        audio_chunks.append(pcm_chunk)

    full_audio = b"".join(audio_chunks)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{video_id}_podcast.wav")

    save_wave_file(output_path, full_audio)

    return output_path


def generate_podcast_from_video(transcript: str, video_id: str, podcast_language_code: str = "en", output_folder: str = "data/audio") -> tuple[str, str]:

    if not transcript:
        raise ValueError("Transcript is empty. Cannot generate podcast.")

    if not video_id:
        raise ValueError("Video ID is empty. Cannot generate podcast.")

    # 1. Let Gemini decide how many speakers (host + specialists) the content needs.
    speakers = plan_speakers(transcript)

    # 2. Summarize the transcript into study notes for the script writer.
    notes = generate_video_notes(transcript, podcast_language_code)

    # 3. Write the dialogue for exactly those speakers.
    podcast_script = generate_multispeaker_podcast_script(notes, speakers, podcast_language_code)

    # 4. Synthesize each line and stitch into one audio file.
    audio_path = generate_podcast_audio(podcast_script, speakers, video_id, output_folder)

    return audio_path, podcast_script

