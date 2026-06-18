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
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "video_ready_for_chat" not in st.session_state:
    st.session_state.video_ready_for_chat = False

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "transcript" not in st.session_state:
    st.session_state.transcript = None
    
if "transcript_language_code" not in st.session_state:
    st.session_state.transcript_language_code = None

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
        "Note / Response Language Code",
        value="en"
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
        st.session_state.video_id = extract_video_id(youtube_url) 
        
        if not st.session_state.video_id:
            st.error("Invalid YouTube URL.")
            st.stop()

        with st.status("Processing video ...", expanded=True) as status:
            with st.spinner("Step 1/3: Fetching transcript..."):
                st.session_state.transcript_language_code = get_transcript_language_code(st.session_state.video_id) 
                st.session_state.transcript = get_transcript(st.session_state.video_id, st.session_state.transcript_language_code)
            
            with st.spinner(f"Step 1.5/3: Translating transcript into {note_language_code} , This may take few moments..."):
                    st.session_state.transcript = translate_transcript_to_language_code(st.session_state.transcript, st.session_state.transcript_language_code, note_language_code)
                    
            with st.spinner("Step 2/3: Generating key topics and notes..."):
                    study_notes_result = generate_video_notes(st.session_state.transcript, note_language_code)
            
            with st.spinner("Step 3/3: Finalizing output..."):
                    st.session_state.notes_result = (
                    study_notes_result
                    )
                    st.success("Video processed successfully!")
            
    elif task_type == "Chat with Video":
            
        st.session_state.video_id = extract_video_id(youtube_url)

        if not st.session_state.video_id:
            st.error("Invalid YouTube URL.")
            st.stop()

        with st.status("Preparing video for chat ...", expanded=True) as status:

            with st.spinner("Step 1/5: Detecting transcript language..."):
                st.session_state.transcript_language_code = get_transcript_language_code(
                    st.session_state.video_id
                )

            with st.spinner("Step 2/5: Fetching transcript..."):
                st.session_state.transcript = get_transcript(
                    st.session_state.video_id,
                    st.session_state.transcript_language_code
                )

            with st.spinner(f"Step 3/5: Translating transcript into {note_language_code}..."):
                st.session_state.transcript = translate_transcript_to_language_code(
                    st.session_state.transcript,
                    st.session_state.transcript_language_code,
                    note_language_code
                )

            with st.spinner("Step 4/5: Splitting transcript into chunks..."):
                st.session_state.chunks = chunk_transcript(
                    st.session_state.video_id,
                    st.session_state.transcript,
                    note_language_code
                )

            with st.spinner("Step 5/5: Creating searchable video knowledge base..."):
                st.session_state.vector_store = store_in_chromadb(
                    st.session_state.chunks,
                    st.session_state.video_id
                )

            st.session_state.video_ready_for_chat = True
            st.session_state.chat_messages = []

            status.update(
                label="Video is ready for chat!",
                state="complete"
            )

        st.success("Video is ready. Ask a question below!")
        
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
    if not st.session_state.video_ready_for_chat:
        st.info("Process a YouTube video first, then ask questions here.")

    else:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask a question about the video")

        if user_question:
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_question
            })

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the video and generating answer..."):
                    relevant_chunks = retrieve_relevant_chunks(
                        user_question,
                        st.session_state.vector_store
                    )

                    answer = generate_rag_answer(
                        user_question,
                        relevant_chunks,
                        note_language_code
                    )

                    st.markdown(answer)

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer
            })
