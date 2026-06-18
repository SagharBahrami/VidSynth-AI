import streamlit as st 
from services import (generate_video_notes, 
                      generate_rag_answer,
                      extract_video_id,
                      get_transcript,
                      chunk_transcript,
                      store_in_chromadb,
                      retrieve_relevant_chunks,
                      translate_transcript_to_language_code,
                      get_transcript_language_code)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="VidSynth AI",
    page_icon="🎬", 
    layout="wide"

)

# -------------------------
# SESSION STATE
# -------------------------

if "notes_result" not in st.session_state:
    st.session_state.notes_result = None


# -------------------------
#SIDEBAR 
# -------------------------

with st.sidebar:
    st.title("🎬VidSynth AI") 
    
    st.markdown("---")
    
    st.write(
        "Transform any YouTube video into keytopics, "
        "study notes, podcast, or a chatbot"
    )
    
    st.subheader("Input Details")
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    note_language_code = st.text_input(
        "Video Language Code",
        placeholder="en"
    )
    
    task_type = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"]

    )
    
    start_button = st.button("✨ Start Processing") 
    
# -------------------------
# MAIN PAGE
# -------------------------


st.title("YouTube Content Synthesizer")
st.write("Paste a video link and select a task from sidebar.")

st.markdown("---")

# -------------------------
# PROCESSING
# -------------------------

if start_button:
    
    if not youtube_url:
        st.warning("Please enter a YouTube URL first.")

    elif task_type == "Notes For You":
        video_id = extract_video_id(youtube_url) 
        
        if not video_id:
            st.error("Invalid YouTube URL.")
            st.stop()

        with st.status("Processing video ...", expanded=True) as status:
            with st.spinner("Step 1/3: Fetching transcript..."):
                transcript_language_code = get_transcript_language_code(video_id) 
                transcript = get_transcript(video_id, transcript_language_code)
            
            with st.spinner(f"Step 1.5/3: Translating transcript into {note_language_code} , This may take few moments..."):
                    transcript = translate_transcript_to_language_code(transcript, transcript_language_code, note_language_code)
                    
            with st.spinner("Step 2/3: Generating key topics and notes..."):
                    study_notes_result = generate_video_notes(transcript, note_language_code)
            
            with st.spinner("Step 3/3: Finalizing output..."):
                    st.session_state.notes_result = (
                    study_notes_result
                    )
                    st.success("Video processed successfully!")
            
    else:
        st.info("Chat with Video will be added after Notes For You works.")
        
# -------------------------
# NOTES OUTPUT
# -------------------------
if task_type == "Notes For You":
    st.subheader("📝 Notes For You")
    
    if st.session_state.notes_result is None:
        st.info("Click Start Processing to generate notes")
    else:
        st.markdown(st.session_state.notes_result)
        

# -------------------------
# CHAT PLACEHOLDER
# -------------------------
elif task_type == "Chat with Video":
    
    st.subheader("💬 Chat with Video")
    st.info("We will build the chat interface after the notes feature.")
