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

def embedding_similarity(questions, retrieved_contexts, method="cosine"):
    """
    Calculates the embedding similarity between reference contexts and the retrieved contexts.
    """
    similarities = []
    if method=="cosine":
        for reference, retrieved in zip(questions, retrieved_contexts):
            ref_embedding = openai_embedding(reference)
            retrieved_embedding = openai_embedding(retrieved)
            similarity = cosine_similarity(ref_embedding, retrieved_embedding)
            similarities.append(round(similarity.item(), 2))
    elif method=="dot":
        for reference, retrieved in zip(questions, retrieved_contexts):
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
    for reference, retrieved in zip(reference_contexts, retrieved_contexts):
        # Find the ROUGE precision between the reference and retrieved contexts
        _rouge_n = rouge_n(retrieved, reference, n=3)
        _rouge_l = rouge_l(retrieved, reference)
        rouge_score = [round(_rouge_n, 2), round(_rouge_l, 2)]
        scores.append(rouge_score)
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
            
    if args.test=="named_entity_score":
        question = ["What is the capital of India?", "What is the capital of France?", "What is the capital of Germany?"]
        context = ["The capital of India is New Delhi", "The capital of France is Paris", "The capital of Germany is Berlin"]
        scores = named_entity_score(question, context)
        print("Named Entity Scores")
        print("-------")
        for i, score in enumerate(scores):
            print(f"Context pair {i+1}|\tNamed Entity Score: {score}")