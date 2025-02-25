import numpy as np
import spacy
from src.utils.algo import lcs

def cosine_similarity(vec_A, vec_B):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec_A, vec_B)
    magnitude_A = np.linalg.norm(vec_A)
    magnitude_B = np.linalg.norm(vec_B)

    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

def n_gram(words, n):
    """
    Generate n-grams from a list of tokens.
    """
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def rouge_n(candidate, reference, n=3):
    """
    Calculate the ROUGE-n score between two sentences.
    """
    ngrams_A, ngrams_B = n_gram(candidate.split(), n), n_gram(reference.split(), n)
    overlaps = len(set(ngrams_A).intersection(set(ngrams_B)))
    recall = overlaps / len(ngrams_B)
    precision = overlaps / len(ngrams_A)
    
    if recall + precision == 0:
        return 0.0
    
    f1_score = (2 * recall * precision) / (recall + precision)
    return f1_score

def rouge_l(candidate, reference):
    """
    Calculate the ROUGE-L score between two sentences.
    """
    candidate_words = candidate.split()
    reference_words = reference.split()

    lcs_length = lcs(candidate_words, reference_words)

    recall = lcs_length / len(reference_words) if reference_words else 0
    precision = lcs_length / len(candidate_words) if candidate_words else 0

    if recall + precision == 0:
        return 0.0

    f1_score = (2 * recall * precision) / (recall + precision)
    return f1_score

def ner(text):
    """
    Extract named entities from a text.
    """
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    return {(ent.text, ent.label_) for ent in doc.ents}