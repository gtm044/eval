import numpy as np
import spacy
from src.utils.algo import lcs

def cosine_similarity(vec_A, vec_B):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec_A: First vector
        vec_B: Second vector
        
    Returns:
        float: Cosine similarity value between 0 and 1, where 1 means identical direction
    """
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec_A, vec_B)
    # Calculate the magnitude (L2 norm) of each vector
    magnitude_A = np.linalg.norm(vec_A)
    magnitude_B = np.linalg.norm(vec_B)

    # Calculate cosine similarity using the formula: dot_product / (||A|| * ||B||)
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

def n_gram(words, n):
    """
    Generate n-grams from a list of tokens.
    
    Args:
        words: List of tokens/words
        n: Size of the n-gram window
        
    Returns:
        list: List of n-grams as strings
    """
    # Create n-grams by sliding a window of size n over the words list
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def rouge_n(candidate, reference, n=3):
    """
    Calculate the ROUGE-n score between two sentences.
    
    Args:
        candidate: Candidate text (e.g., generated summary)
        reference: Reference text (e.g., ground truth)
        n: Size of n-grams to use (default: 3)
        
    Returns:
        float: F1 score representing the ROUGE-n metric
    """
    # Generate n-grams for both candidate and reference texts
    ngrams_A, ngrams_B = n_gram(candidate.split(), n), n_gram(reference.split(), n)
    # Overlapping n-grams
    overlaps = len(set(ngrams_A).intersection(set(ngrams_B)))
    # Calculate recall: overlaps / total reference n-grams
    recall = overlaps / len(ngrams_B) if len(ngrams_B) > 0 else 0
    # Calculate precision: overlaps / total candidate n-grams
    precision = (overlaps / len(ngrams_A)) if len(ngrams_A) > 0 else 0
    
    # Return 0 if both recall and precision are 0
    if recall + precision == 0:
        return 0.0
    
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1_score = (2 * recall * precision) / (recall + precision)
    return f1_score

def rouge_l(candidate, reference):
    """
    Calculate the ROUGE-L score between two sentences using longest common subsequence.
    
    Args:
        candidate: Candidate text (e.g., generated summary)
        reference: Reference text (e.g., ground truth)
        
    Returns:
        float: F1 score representing the ROUGE-L metric
    """
    # Split texts into words
    candidate_words = candidate.split()
    reference_words = reference.split()

    # Calculate length of longest common subsequence
    lcs_length = lcs(candidate_words, reference_words)

    # Calculate recall: LCS length / reference length
    recall = lcs_length / len(reference_words) if reference_words else 0
    # Calculate precision: LCS length / candidate length
    precision = lcs_length / len(candidate_words) if candidate_words else 0

    # Return 0 if both recall and precision are 0
    if recall + precision == 0:
        return 0.0

    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1_score = (2 * recall * precision) / (recall + precision)
    return f1_score

def ner(text):
    """
    Extract named entities from a text using spaCy.
    
    Args:
        text: Input text to analyze
        
    Returns:
        set: Set of tuples containing (entity_text, entity_label)
    """
    # Load the English language model (medium size)
    nlp = spacy.load("en_core_web_md")
    # Process the text with spaCy
    doc = nlp(text)
    # Return set of named entities with their labels
    return {(ent.text, ent.label_) for ent in doc.ents}