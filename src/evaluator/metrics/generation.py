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

def llm_grading(queries: List[str], ground_truths: List[str], model_answers: List[str], rubrics: Optional[List[str]] = None):
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
    for query, ground_truth, model_answer, rubric in zip(queries, ground_truths, model_answers, rubrics):
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
        "What are the key factors that contribute to climate change?",
        "Explain the process of machine learning and its applications in healthcare.",
        "What were the main causes of World War II?",
        "Describe the structure and function of DNA in human cells.",
        "What are the economic implications of implementing a universal basic income?"
    ]
    ground_truths = [
        "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, deforestation, industrial processes, and agricultural practices. These activities increase atmospheric CO2 and other greenhouse gases, trapping heat and raising global temperatures.",
        "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming. In healthcare, it's used for disease diagnosis, treatment recommendation, patient monitoring, drug discovery, and medical image analysis.",
        "World War II was caused by multiple factors including the harsh Treaty of Versailles, the Great Depression, the rise of fascism and militarism in Germany, Italy, and Japan, appeasement policies by Western powers, and Hitler's aggressive expansionism.",
        "DNA is a double-helix structure composed of nucleotides containing phosphate groups, deoxyribose sugar, and nitrogenous bases. It stores genetic information, replicates during cell division, and directs protein synthesis through transcription and translation processes.",
        "Implementing universal basic income could reduce poverty and income inequality, provide economic security, stimulate consumer spending, and support entrepreneurship. However, it may increase inflation, reduce work incentives, and require significant tax increases or budget reallocations."
    ]
    model_answers = [
        "Climate change is mainly caused by human activities that release greenhouse gases into the atmosphere, particularly from burning fossil fuels like coal and oil. Deforestation and industrial processes also contribute significantly.",
        "Machine learning involves training algorithms on data to identify patterns and make predictions. In healthcare, it's revolutionizing diagnosis through image analysis, personalizing treatment plans, predicting patient outcomes, and accelerating drug discovery.",
        "The primary causes of World War II included economic hardship from the Great Depression, the rise of totalitarian regimes, and territorial ambitions of Nazi Germany. The failure of the League of Nations and appeasement policies also played important roles.",
        "DNA is a molecule that carries genetic instructions for development and functioning of all living organisms. It has a double-helix structure made of nucleotides, with complementary base pairs forming the rungs of the ladder-like structure.",
        "A universal basic income would provide regular payments to all citizens regardless of employment status. This could reduce poverty and provide economic security, but might decrease workforce participation and would require significant government funding."
    ]
    rubrics = [
        "Evaluate the answer based on scientific accuracy, comprehensiveness of causes mentioned, and explanation of the greenhouse effect mechanism. Higher scores for answers that include both human and natural factors with appropriate emphasis on human contributions.",
        "Grade based on technical accuracy of ML explanation, breadth of healthcare applications mentioned, and depth of explanation about how ML improves healthcare outcomes. Consider whether limitations and ethical considerations are addressed.",
        "Assess the answer on historical accuracy, comprehensiveness of causes (political, economic, social factors), chronological understanding, and balanced perspective on different nations' roles in the conflict.",
        "Evaluate based on structural accuracy (nucleotides, base pairs, double helix), functional explanation (genetic code, protein synthesis), and clarity of the relationship between structure and function.",
        "Grade based on balanced coverage of both potential benefits and drawbacks, economic theory application, consideration of implementation challenges, and discussion of real-world examples or pilot programs."
    ]
    print(llm_grading(queries, ground_truths, model_answers, rubrics))
    