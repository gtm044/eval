from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize
import nltk

load_dotenv()

nltk.download('punkt_tab')

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

def qa_bm25(text, questions):
    """
    Get the answer to the question using the BM25 model.
    """
    sentences = sent_tokenize(text)
    tokenized_corpus = [nltk.word_tokenize(sent) for sent in sentences]
    bm25 = BM25Okapi(tokenized_corpus)
    
    query = nltk.word_tokenize(questions)
    best_match_index = bm25.get_top_n(query, sentences, n=1)
    if best_match_index:
        return best_match_index[0]
    
    return None