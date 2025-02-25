# Implement the following metrics
# - Context Reelvance
# - Context Recall
# - Context Precision
# - Embedding Similarity
# - Redundancy
# - Named Entity Match
# - Keyword Overlap Score

from openai import OpenAI
import numpy as np

def get_embedding(text):
    """
    Get the embedding of the text using the OpenAI API.
    """
    openai = OpenAI()
    response = openai.embeddings.create(
        input = text,
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(vec_A, vec_B):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec_A, vec_B)
    magnitude_A = np.linalg.norm(vec_A)
    magnitude_B = np.linalg.norm(vec_B)

    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

def embedding_similarity(reference_contexts, retrieved_contexts, method="cosine"):
    """
    Calculates the cosine similarity between reference contexts and the retrieved contexts.
    """
    similarities = []
    if method=="cosine":
        for reference, retrieved in zip(reference_contexts, retrieved_contexts):
            ref_embedding = get_embedding(reference)
            retrieved_embedding = get_embedding(retrieved)
            similarity = cosine_similarity(ref_embedding, retrieved_embedding)
            similarities.append(similarity)
    elif method=="dot":
        for reference, retrieved in zip(reference_contexts, retrieved_contexts):
            ref_embedding = get_embedding(reference)
            retrieved_embedding = get_embedding(retrieved)
            similarity = np.dot(ref_embedding, retrieved_embedding)
            similarities.append(similarity)
    return similarities

if __name__=='__main__':
    reference_contexts = [
        "This is a test document. It has a few sentences.",
        "This is another test document. It also has a few sentences.",
        "My name is Goutham"
    ]
    retrieved_contexts = [
        "This is a test document. It has a few sentences.",
        "This is another test document. It also has a few sentences.",
        "My name is Goutham"
    ]
    print(embedding_similarity(reference_contexts, retrieved_contexts, method="cosine"))