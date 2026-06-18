RAG_PROMPT = """ You are a helpful chatbot answering questions about a YouTube video.

Use only the transcript context below.

Transcript context:
{context}

User question:
{question}

Answer clearly:
"""
NOTES_PROMPT = """
You are an AI study assistant.

Create organized study notes from the YouTube transcript below.

Write the entire response in this language code:
{note_language_code}

Rules:
- Use only the transcript.
- Do not add outside information.
- Do not include an introduction like "Here are your notes".
- Keep the notes clear, concise, and useful for studying.

Include exactly these sections. 
Also translate the header of each section to {note_language_code} 
and based on {note_language_code} decide if your response should be left to right or right to left
(for example farsi is written right to left and english is written left to right):

### Short Summary
Write 2-3 sentences.

### Key Topics
- Topic 1
- Topic 2
- Topic 3

### Content Outline
- Main section 1
- Main section 2
- Main section 3

### Detailed Study Notes
- Important note 1
- Important note 2
- Important note 3

### Important Takeaways
- Takeaway 1
- Takeaway 2
- Takeaway 3

Transcript:
{context}
"""

TRANSLATION_PROMPT = """
You are a professional translator.

Translate the transcript below into clear {note_language_code}.

Rules:
- Do not summarize.
- Do not add extra information.
- Keep the meaning accurate.
- Preserve the order of ideas.
- Return only the {note_language_code} translation.

Original language code:
{transcript_language_code}

Transcript:
{transcript}
"""