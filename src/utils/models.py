from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def openai_embedding(text):
    """
    Get the embedding of the text using the OpenAI API.
    """
    openai = OpenAI()
    response = openai.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding