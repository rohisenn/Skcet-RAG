from groq import Groq
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt):
    """Non-streaming call – used for follow-up generation only."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def stream_groq(prompt):
    """Streaming generator – yields text tokens one-by-one for live display."""
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
