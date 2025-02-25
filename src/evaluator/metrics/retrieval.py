# Implement the following metrics
# - Context Reelvance
# - Context Recall
# - Context Precision
# - Embedding Similarity
# - Redundancy
# - Named Entity Match
# - Keyword Overlap Score

from src.utils.models import openai_embedding
from src.utils.nlp import cosine_similarity, n_gram, rouge_n, rouge_l
import argparse
import numpy as np

def embedding_similarity(reference_contexts, retrieved_contexts, method="cosine"):
    """
    Calculates the cosine similarity between reference contexts and the retrieved contexts.
    """
    similarities = []
    if method=="cosine":
        for reference, retrieved in zip(reference_contexts, retrieved_contexts):
            ref_embedding = openai_embedding(reference)
            retrieved_embedding = openai_embedding(retrieved)
            similarity = cosine_similarity(ref_embedding, retrieved_embedding)
            similarities.append(round(similarity.item(), 2))
    elif method=="dot":
        for reference, retrieved in zip(reference_contexts, retrieved_contexts):
            ref_embedding = openai_embedding(reference)
            retrieved_embedding = openai_embedding(retrieved)
            similarity = np.dot(ref_embedding, retrieved_embedding)
            similarities.append(round(similarity.item(), 2))
    return similarities

def context_score(reference_contexts, retrieved_contexts):
    """
    Calculates context precision, recall , f1 score between the reference and retrieved contexts.
    Returns a tuple of (precision, recall, f1 score)
    """
    scores = []
    for reference, retrieved in zip(reference_contexts, retrieved_contexts):
        # Find the ROUGE precision between the reference and retrieved contexts
        _rouge_n = rouge_n(retrieved, reference, n=3)
        _rouge_l = rouge_l(retrieved, reference)
        rouge_score = [round(_rouge_n, 2), round(_rouge_l, 2)]
        scores.append(rouge_score)
    return scores
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=str)
    parser.add_argument("method", type=str)
    args = parser.parse_args()
    reference_contexts = [
        "This is a test document. It has a few sentences.",
        "This is another test document. It does not have a few sentences.",
        "I live in Bengaluru"
    ]
    retrieved_contexts = [
        "This is a test document. It has a few sentences.",
        "This is another test document. It also has a few sentences.",
        "My name is Bob"
    ]
    if args.test=="embedding_similarity":
        if args.method=="cosine":
            print(embedding_similarity(reference_contexts, retrieved_contexts, method="cosine"))
        elif args.method=="dot":
            print(embedding_similarity(reference_contexts, retrieved_contexts, method="dot"))
    
    if args.test=="context_score":
        scores = context_score(reference_contexts, retrieved_contexts)
        print("ROUGE_3, ROUGE_L F1 Scores")
        print("-------")
        for i, score in enumerate(scores):
            print(f"Context pair {i+1}|\tROUGE_3: {score[0]}\tROUGE_L: {score[1]}")