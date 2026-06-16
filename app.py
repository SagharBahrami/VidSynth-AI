import streamlit as st 
from services import generate_video_notes, generate_rag_answer

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
    
    language_code = st.text_input(
        "Video Language Code",
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
        with st.status("Processing video ...", expanded=True) as status:
            
            st.write("Step 1/3: Fetching transcript...")

            if language_code.lower() != "en":
                st.write("Step 1.5/3: Translating transcript into English...")
            
            st.write("Step 2/3: Generating key topics and notes...")
            st.write("Step 3/3: Finalizing output...")

        
            st.session_state.notes_result = generate_video_notes(
                youtube_url,
                language_code
            )

            status.update(
                label="Video processed successfully!",
                state="complete"
            )

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
