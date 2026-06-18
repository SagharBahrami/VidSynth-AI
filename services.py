from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from config import (GEMINI_MODEL, 
                    CHUNK_OVERLAP, 
                    CHUNK_SIZE, 
                    TOP_K,
                    CHROMA_DB_PATH,
                    EMBEDDING_MODEL, 
                    GEMINI_API_KEY
                    )

from prompts import RAG_PROMPT, NOTES_PROMPT, TRANSLATION_PROMPT


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY
)

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


def retrieve_relevant_chunks(question: str, vector_store: Chroma) -> list[Document]:
    relevant_chunks = vector_store.similarity_search(
        question,
        k=TOP_K
    )
    return relevant_chunks


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

