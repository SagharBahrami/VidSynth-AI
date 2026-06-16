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

from prompts import RAG_PROMPT, NOTES_PROMPT

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


def get_transcript_from_url(url: str, language_code: str) -> str:
    
    video_id = extract_video_id(url)
    
    if not video_id:
       raise ValueError("Not a valid url")
    
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=[language_code])
    transcript_parts = []
    for snippet in fetched_transcript:
        transcript_parts.append(snippet.text)
    transcript = " ".join(transcript_parts)
    
    return video_id, transcript

def chunk_transcript(url: str, language_code: str) -> list[Document]:
    
    video_id, transcript = get_transcript_from_url(url, language_code)
    document = Document(
        page_content=transcript,
        metadata={
            "video_id": video_id,
            "language_code": language_code
        }
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents([document])
    
    for index, chunk in enumerate(chunks):
        
        chunk.metadata["chunk_index"] = index

    return video_id, chunks

def store_in_chromadb(url: str, language_code: str) -> Chroma:
    
    video_id, chunks = chunk_transcript(url, language_code)
    
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

def generate_rag_answer(question, relevant_chunks: list[Document]) -> str:
    
    if not question:
        raise ValueError("Question is empty")
    
    if not relevant_chunks:
        raise ValueError("No relevant chunks found")
    context = "\n\n".join(
        chunk.page_content for chunk in relevant_chunks
    )
    
    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )
    
    response = llm.invoke(prompt)
    
    return extract_response_text(response)

def generate_video_notes(url: str, language_code: str) ->str:
    video_id, transcript = get_transcript_from_url(url, language_code)
    
    
    prompt = NOTES_PROMPT.format(
        context=transcript
    )
    
    response = llm.invoke(prompt)
    
    return extract_response_text(response)