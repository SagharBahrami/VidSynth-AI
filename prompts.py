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

Include these sections:

1. Short Summary
2. Key Topics
3. Content Outline
4. Detailed Study Notes
5. Important Takeaways

Transcript:
{context}

Write the notes clearly and professionally.
"""