# Implement the following metrics:
# - BLEU score
# - ROGUE score - Done
# - Faithfulness
# - Answer Relevancy
# - Factual Correctness
# - Contradiction Detection Score - If time permits
# - LLM as a judge - If time permits
import numpy as np
from src.utils.nlp import rouge_n, rouge_l, cosine_similarity
from src.utils.models import openai_embedding
from nltk.translate.bleu_score import sentence_bleu
from keybert import KeyBERT

def rouge_score(answers, responses):
    """
    Calculates context precision, recall , f1 score between the LLM responses and the ground truth answer.
    Returns the F1 score for ROUGE-N and ROUGE-L
    """
    scores = []
    for reference, retrieved in zip(answers, responses):
        # Find the ROUGE precision between the reference and retrieved contexts
        _rouge_n = rouge_n(retrieved, reference, n=3)
        _rouge_l = rouge_l(retrieved, reference)
        rouge_score = [round(_rouge_n, 2), round(_rouge_l, 2)]
        scores.append(rouge_score)
    return scores

def bleu_score(answers, responses):
    """
    Calculate the BLEU score between the LLM responses and the ground truth answer.
    """
    scores = []
    for reference, retrieved in zip(answers, responses):
        score = sentence_bleu([reference], retrieved)
        scores.append(round(score, 2))
    return scores

def faithfulness(retrieved_contexts, responses):
    """
    Calculates the overlap keywords between the retrieved contexts and the llm responses.
    """
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    scores = []
    for _context, _responses in zip(retrieved_contexts, responses):
        c_keywords = model.extract_keywords(_context)
        r_keywords = model.extract_keywords(_responses)
        # output is a list of tuples (keyword, score), extract the keywords
        c_keywords = [keyword for keyword, score in c_keywords]
        r_keywords = [keyword for keyword, score in r_keywords]
        # Calculate the overlap
        overlap = len(set(c_keywords).intersection(set(r_keywords)))
        faithfulness_score = round((overlap / len(r_keywords)) * 100, 2)
        scores.append(faithfulness_score)
    return scores 

def response_similarity(answers, responses, method="cosine"):
    """
    Calculates the embedding similarity between the llm responses and the ground truth answers.
    """
    similarities = []
    if method=="cosine":
        for _answers, _responses in zip(answers, responses):
            answer_embedding = openai_embedding(_answers)
            response_embedding = openai_embedding(_responses)
            similarity = cosine_similarity(answer_embedding, response_embedding)
            similarities.append(round(similarity.item(), 2))
    elif method=="dot":
        for _answers, _responses in zip(answers, responses):
            answer_embedding = openai_embedding(_answers)
            response_embedding = openai_embedding(_responses)
            similarity = np.dot(answer_embedding, response_embedding)
            similarities.append(round(similarity.item(), 2))
    return similarities
        

if __name__ == '__main__':
    answers = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog"
    ]
    responses = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "India is the capital of new delhi and dog jumps"
    ]
    print(rouge_score(answers, responses))
    print(bleu_score(answers, responses))
    print(faithfulness(answers, responses))