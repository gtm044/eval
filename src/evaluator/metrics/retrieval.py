# Implement the following metrics
# - Context Reelvance
# - Context Recall
# - Context Precision
# - Embedding Similarity
# - Redundancy
# - Named Entity Match
# - Accuracy (total )

from src.utils.models import openai_embedding
from src.utils.nlp import cosine_similarity, rouge_n, rouge_l, ner
import argparse
import numpy as np

def context_similarity(reference_contexts, retrieved_contexts, method="cosine"):
    """
    Calculates the embedding similarity between reference contexts and the retrieved contexts.
    """
    similarities = []
    # Retrieved contexts are a list of list of strings, make it a list of strings, concatenate strings in each list
    retrieved_contexts = [' '.join(context) for context in retrieved_contexts]
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
    Returns the F1 score for ROUGE-N and ROUGE-L
    """
    scores = []
    # Retrieved contexts are a list of list of strings, make it a list of strings, concatenate strings in each list
    retrieved_contexts = [' '.join(context) for context in retrieved_contexts]
    for reference, retrieved in zip(reference_contexts, retrieved_contexts):
        # Find the ROUGE precision between the reference and retrieved contexts
        _rouge_n = rouge_n(retrieved, reference, n=3)
        _rouge_l = rouge_l(retrieved, reference)
        # rouge_score = [round(_rouge_n, 2), round(_rouge_l, 2)]
        scores.append(_rouge_l)
    return scores


def named_entity_score(question, retrieved_contexts):
    """
    Calculate the named entity score between the reference question and the retrieved contexts.
    """
    scores = []
    for question, context in zip(question, retrieved_contexts):
        question_entities = ner(question)
        context_entities = ner(context)
        intersection = len(question_entities.intersection(context_entities))
        union = len(question_entities.union(context_entities))
        score = intersection / union # How to convert the intersection to a score? -> Probably get the number of interections/total number of entities in the question * 100?
        scores.append(round(score, 2))
    return scores

def retrieval_accuracy(reference_contexts, retrived_contexts):
    """
    Calculate the accuracy of the retrieved contexts.
    """
    accuracy = 0
    for reference, retrieved in zip(reference_contexts, retrived_contexts):
        if reference == retrieved:
            accuracy += 1
    return round(accuracy/len(reference_contexts), 2)

# Function attributes
context_similarity.name = "context_similarity"
context_score.name = "context_score"
named_entity_score.name = "named_entity_score"
            
            
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
            print(context_similarity(reference_contexts, retrieved_contexts, method="cosine"))
        elif args.method=="dot":
            print(context_similarity(reference_contexts, retrieved_contexts, method="dot"))
    
    if args.test=="context_score":
        scores = context_score(reference_contexts, retrieved_contexts)
        print("ROUGE_3, ROUGE_L F1 Scores")
        print("-------")
        for i, score in enumerate(scores):
            print(f"Context pair {i+1}|\tROUGE_3: {score[0]}\tROUGE_L: {score[1]}")
            
    if args.test=="named_entity_score":
        question = ["What is the capital of India?", "What is the capital of France?", "What is the capital of Germany?"]
        context = ["The capital of India is New Delhi", "The capital of France is Paris", "The capital of Germany is Berlin"]
        scores = named_entity_score(question, context)
        print("Named Entity Scores")
        print("-------")
        for i, score in enumerate(scores):
            print(f"Context pair {i+1}|\tNamed Entity Score: {score}")