import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from rank_bm25 import BM25Okapi
from src.utils.prompts import llm_judge_prompt
# from nltk.tokenize import sent_tokenize
# import nltk

load_dotenv()

# nltk.download('punkt_tab')

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

# def qa_bm25(text, questions):
#     """
#     Get the answer to the question using the BM25 model.
#     """
#     sentences = sent_tokenize(text)
#     tokenized_corpus = [nltk.word_tokenize(sent) for sent in sentences]
#     bm25 = BM25Okapi(tokenized_corpus)
    
#     query = nltk.word_tokenize(questions)
#     best_match_index = bm25.get_top_n(query, sentences, n=1)
#     if best_match_index:
#         return best_match_index[0]
    
#     return None

# IMplement the llm as a judge for the answer faithfulness in agentic va;idatipn metrics
def llm_as_a_judge(answer, context_reference):
    """
    Use an llm to judge the answer faithfulness.
    """
    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Convert the context-reference to a json string
    context_ref_string = json.dumps(context_reference)
    
    messages = [
        {
            "role": "system",
            "content": llm_judge_prompt
        },
        {
            "role": "user",
            "content": f"Answer: {answer}\n\nContext Reference: {context_ref_string}"
        }
    ]
    
    completion = openai_client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
    )
    
    response = completion.choices[0].message.content
    
    # Process the response json string
    response = response.replace("```json", "").replace("```", "")
    
    # Convert the string to a json object
    response = json.loads(response)
    return response["faithfulness_score"]
    
    
if __name__=='__main__':
    answer = "The current price of copper is $0.0098 per gram."
    reference = "0.0098"
    
    llm_as_a_judge(answer, reference)
    

