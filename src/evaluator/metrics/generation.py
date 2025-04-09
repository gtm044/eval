import numpy as np
import os
from src.utils.nlp import rouge_n, rouge_l, cosine_similarity
from src.utils.models import openai_embedding
from nltk.translate.bleu_score import sentence_bleu
from keybert import KeyBERT
from typing import List, Optional
from openai import OpenAI

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
        faithfulness_score = round((overlap / len(c_keywords)) * 100, 2)
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

def llm_grading(queries: List[str], ground_truths: List[str], model_answers: List[str], rubric: Optional[str] = ""):
    """
    Use an llm to grade the model answer.
    """
    EVALUATION_PROMPT = """
    Your job is to evaluate the performance of an AI-powered question answering system. You will be given a query, a ground truth answer, and the answer given by the AI. Your task is to grade the AI's answer on a scale of 0-10. A score of 0 means the AI's answer is wrong. A score of 10 means the AI's answer is completely correct.

    Your response must ONLY be an integer between 0 and 10 (inclusive). Do not include any other text in your response.

    GUIDELINES FOR GRADING
    - The ground truth answers are often lacking in detail, so if the AI's answer is more detailed than the ground truth answer, then that's generally a good sign.
    - Be wary of overly broad or general AI answers. If the AI's answer lacks specifics, then it probably isn't a good answer.
    - If a grading rubric is included in the GRADING RUBRIC section, then pay close attention to it. The rubric will tell you specific things to look for in the AI's answer.
    - Maintain high standards when grading. A score of 10 should be reserved for answers that are nearly perfect. Answers that miss key details or don't fully answer the question should be heavily penalized.

    QUERY
    {query}

    GROUND TRUTH ANSWER
    {ground_truth_answer}

    GRADING RUBRIC
    {rubric}

    AI-GENERATED ANSWER
    {model_answer}

    GRADE
    """.strip()
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    grades = []
    for query, ground_truth, model_answer in zip(queries, ground_truths, model_answers):
        prompt = EVALUATION_PROMPT.format(query=query, ground_truth_answer=ground_truth, rubric=rubric, model_answer=model_answer)
        chat_messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = chat_messages,
            temperature = 0.0,
        )
        grade_output = response.choices[0].message.content.strip()
        grade = int(grade_output)
        grades.append(grade)
    return grades


llm_grading.name = "llm_grading"


if __name__ == '__main__':
    # answers = [
    #     "The quick brown fox jumps over the lazy dog",
    #     "The quick brown fox jumps over the lazy dog",
    #     "The quick brown fox jumps over the lazy dog"
    # ]
    # responses = [
    #     "The quick brown fox jumps over the lazy dog",
    #     "The quick brown fox jumps over the lazy dog",
    #     "India is the capital of new delhi and dog jumps"
    # ]
    # print(rouge_score(answers, responses))
    # print(bleu_score(answers, responses))
    # print(faithfulness(answers, responses))
    
    queries = [
        "What is the capital of India?",
        "Who is the president of the United States?",
        "What is the capital of the moon?"
    ]
    ground_truths = [
        "New Delhi",
        "Joe Biden",
        "London"
    ]
    model_answers = [
        "New Delhi",
        "Joe Jilden",
        "London"
    ]
    print(llm_grading(queries, ground_truths, model_answers))