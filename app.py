import streamlit as st 
from services import (generate_video_notes, 
                      generate_rag_answer,
                      extract_video_id,
                      get_transcript,
                      chunk_transcript,
                      store_in_chromadb,
                      retrieve_relevant_chunks,
                      translate_transcript_to_language_code,
                      get_transcript_language_code,
                      generate_audio_from_text,
                      convert_audio_to_text,
                      generate_podcast_from_video)

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

if "notes_audio_path" not in st.session_state:
    st.session_state.notes_audio_path = None
    
if "voice_mode" not in st.session_state:
    st.session_state.voice_mode = False

if "processed_audio_hash" not in st.session_state:
    st.session_state.processed_audio_hash = None

if "podcast_audio_path" not in st.session_state:
    st.session_state.podcast_audio_path = None

if "podcast_script" not in st.session_state:
    st.session_state.podcast_script = None
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
        ["Chat with Video", "Notes For You", "Podcast"]

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
                st.session_state.notes_audio_path = None
                study_notes_result = generate_video_notes(st.session_state.transcript, note_language_code)
            
            with st.spinner("Step 3/3: Finalizing output..."):
                st.session_state.notes_result = (
                        study_notes_result
                    )
                st.success("Video processed successfully!")
                    
                
    elif task_type == "Podcast":
        st.session_state.video_id = extract_video_id(youtube_url)

        if not st.session_state.video_id:
            st.error("Invalid YouTube URL.")
            st.stop()

        with st.status("Generating podcast ...", expanded=True) as status:
            with st.spinner("Step 1/4: Fetching transcript..."):
                st.session_state.transcript_language_code = get_transcript_language_code(st.session_state.video_id)
                st.session_state.transcript = get_transcript(st.session_state.video_id, st.session_state.transcript_language_code)

            with st.spinner(f"Step 2/4: Translating transcript into {note_language_code}..."):
                st.session_state.transcript = translate_transcript_to_language_code(st.session_state.transcript, st.session_state.transcript_language_code, note_language_code)

            with st.spinner("Step 3/4: Writing the podcast script..."):
                st.session_state.podcast_audio_path = None
                st.session_state.podcast_script = None

            with st.spinner("Step 4/4: Generating multi-speaker audio (this may take a moment)..."):
                podcast_audio_path, podcast_script = generate_podcast_from_video(
                    st.session_state.transcript,
                    st.session_state.video_id,
                    note_language_code
                )
                st.session_state.podcast_audio_path = podcast_audio_path
                st.session_state.podcast_script = podcast_script

            status.update(label="Podcast generated successfully!", state="complete")

        st.success("Podcast is ready below!")

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
        if st.button("🎧 Generate Audio Version"):
            try:
                with st.spinner("Generating audio version..."):
                    st.session_state.notes_audio_path, audio_language_code = generate_audio_from_text(
                        st.session_state.notes_result,
                        note_language_code,
                        st.session_state.video_id,
                        "notes"
                    )
                if audio_language_code != note_language_code:
                    st.info(f"Audio is not supported for '{note_language_code}', so English audio was generated instead.")
                st.success("Audio generated successfully!")
            except Exception as error:
                st.error(error)

        if st.session_state.notes_audio_path:
            with open(st.session_state.notes_audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            st.audio(audio_bytes, format="audio/mp3")   
        
        st.markdown("---")
             
        st.markdown(st.session_state.notes_result)
        



        

# -------------------------
# CHAT PLACEHOLDER
# -------------------------
# elif task_type == "Chat with Video":
    
#     st.subheader("💬 Chat with Video")
#     if not st.session_state.video_ready_for_chat:
#         st.info("Process a YouTube video first, then ask questions here.")

#     else:
#         #Show previous chat messages
#         for message in st.session_state.chat_messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         user_question = st.chat_input("Ask a question about the video")

#         if user_question:
#             st.session_state.chat_messages.append({
#                 "role": "user",
#                 "content": user_question
#             })

#             with st.chat_message("user"):
#                 st.markdown(user_question)

#             with st.chat_message("assistant"):
#                 with st.spinner("Searching the video and generating answer..."):
#                     relevant_chunks = retrieve_relevant_chunks(
#                         user_question,
#                         st.session_state.vector_store
#                     )

#                     answer = generate_rag_answer(
#                         user_question,
#                         relevant_chunks,
#                         note_language_code
#                     )

#                     st.markdown(answer)

#             st.session_state.chat_messages.append({
#                 "role": "assistant",
#                 "content": answer
#             })
elif task_type == "Podcast":
    st.subheader("🎙️ Podcast")

    if st.session_state.podcast_audio_path is None:
        st.info("Click Start Processing to generate a podcast from the video.")
    else:
        with open(st.session_state.podcast_audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/wav")

        st.download_button(
            "⬇️ Download Podcast",
            data=audio_bytes,
            file_name=f"{st.session_state.video_id}_podcast.wav",
            mime="audio/wav"
        )

        if st.session_state.podcast_script:
            with st.expander("📄 View podcast script"):
                st.markdown(st.session_state.podcast_script)

elif task_type == "Chat with Video":

    st.subheader("💬 Chat with Video")

    if not st.session_state.video_ready_for_chat:
        st.info("Process a YouTube video first, then ask questions here.")

    else:
        # Show previous chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("audio_path"):
                    with open(message["audio_path"], "rb") as audio_file:
                        audio_bytes = audio_file.read()

                    st.audio(audio_bytes, format="audio/mp3")

        # Voice mode button
        if st.button("🎙️ Ask by Voice"):
            st.session_state.voice_mode = True

        user_question = None

        # Voice input area
        if st.session_state.voice_mode:
            audio_question = st.audio_input("Record your question")

            if audio_question:
                audio_bytes = audio_question.getvalue()
                audio_hash = hash(audio_bytes)

                if st.session_state.processed_audio_hash != audio_hash:
                    try:
                        with st.spinner("Converting audio to text..."):
                            user_question = convert_audio_to_text(
                                audio_question,
                                note_language_code
                            )

                        st.session_state.processed_audio_hash = audio_hash
                        st.success(f"You asked: {user_question}")

                    except Exception as error:
                        st.error(error)

        # Typed input area
        typed_question = st.chat_input("Ask a question about the video")

        if typed_question:
            user_question = typed_question

        # Process either typed question or voice question
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

                    answer_audio_path = None

                    try:
                        answer_audio_path, audio_language_code = generate_audio_from_text(
                            answer,
                            note_language_code,
                            st.session_state.video_id,
                            "chat_answer"
                        )

                        with open(answer_audio_path, "rb") as audio_file:
                            answer_audio_bytes = audio_file.read()

                        st.audio(answer_audio_bytes, format="audio/mp3")

                    except Exception as error:
                        st.warning(f"Audio answer was not generated: {error}")

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "audio_path": answer_audio_path
            })

            st.session_state.voice_mode = False
