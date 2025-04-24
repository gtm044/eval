import numpy as np
import os
from src.utils.nlp import rouge_n, rouge_l, cosine_similarity, ner
from src.utils.models import openai_embedding
from nltk.translate.bleu_score import sentence_bleu
from keybert import KeyBERT
from typing import List, Optional
from openai import OpenAI
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import json
import re

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


def hybrid_faithfulness(retrieved_contexts, responses):
    """
    A hybrid approach to faithfulness combining sentence embeddings, entity extraction,
    and numeric fact checking. Penalizes contradicting information, not additional content.
    
    Args:
        retrieved_contexts (List[str]): List of reference contexts
        responses (List[str]): List of LLM-generated responses
        
    Returns:
        List[float]: Faithfulness scores between 0 and 1
    """
    scores = []
    
    for context, response in zip(retrieved_contexts, responses):
        if not context or not response:
            scores.append(0.0)
            continue
            
        try:
            context_sentences = sent_tokenize(context)
            response_sentences = sent_tokenize(response)
        except:
            nltk.download('punkt')
            context_sentences = sent_tokenize(context)
            response_sentences = sent_tokenize(response)
            
        if not response_sentences:
            scores.append(0.0)
            continue
            
        context_embeddings = [openai_embedding(sent) for sent in context_sentences]
        response_embeddings = [openai_embedding(sent) for sent in response_sentences]
        
        similarity_scores = []
        for ctx_emb in context_embeddings:
            similarities = [cosine_similarity(ctx_emb, resp_emb) for resp_emb in response_embeddings]
            max_similarity = max(similarities) if similarities else 0
            similarity_scores.append(max_similarity)
            
        avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        context_entities = ner(context)
        response_entities = ner(response)
        
        context_entity_types = {entity_type for _, entity_type in context_entities}
        relevant_response_entities = {(entity, entity_type) for entity, entity_type in response_entities 
                                     if entity_type in context_entity_types}
        
        contradicting_entities = 0
        if relevant_response_entities:
            for resp_entity, resp_type in relevant_response_entities:
                similar_context_entities = {entity for entity, entity_type in context_entities if entity_type == resp_type}
                
                if similar_context_entities and resp_entity not in [entity for entity, entity_type in context_entities]:
                    has_similar = False
                    for ctx_entity in similar_context_entities:
                        if (resp_entity in ctx_entity or ctx_entity in resp_entity or
                            cosine_similarity(openai_embedding(resp_entity), openai_embedding(ctx_entity)) > 0.85):
                            has_similar = True
                            break
                    
                    if not has_similar:
                        contradicting_entities += 1
            
            entity_score = 1.0 - (contradicting_entities / len(relevant_response_entities))
        else:
            entity_score = 1.0
            
        context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', context))
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', response))
        
        numeric_score = 1.0
        if context_numbers and response_numbers:
            numeric_score = 0.9
        
        final_score = (0.7 * avg_similarity_score) + (0.2 * entity_score) + (0.1 * numeric_score)
        scores.append(round(final_score, 3))
        
    return scores

def contradiction_faithfulness(retrieved_contexts, responses):
    """
    A specialized faithfulness metric focused on detecting contradictions rather than
    penalizing additional information. Based on the paper "SUMMAC: Re-Visiting NLI-based
    Models for Inconsistency Detection in Summarization" (Laban et al., 2022).
    
    This approach only penalizes statements that actively contradict the reference context,
    allowing responses to include additional information freely as long as it doesn't
    conflict with the provided context.
    
    Args:
        retrieved_contexts (List[str]): List of reference contexts
        responses (List[str]): List of LLM-generated responses
        
    Returns:
        List[float]: Faithfulness scores between 0 and 1, where 1 means no contradictions
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    scores = []
    
    for context, response in zip(retrieved_contexts, responses):
        # Skip empty pairs
        if not context or not response:
            scores.append(1.0)  # No contradictions possible with empty input
            continue
            
        # Break response into sentences
        try:
            response_sentences = sent_tokenize(response)
        except:
            nltk.download('punkt')
            response_sentences = sent_tokenize(response)
            
        if not response_sentences:
            scores.append(1.0)  # No contradictions with empty response
            continue
            
        # Only check sentences that are likely to contain factual claims
        # Filter out very short sentences and questions
        factual_sentences = [s for s in response_sentences 
                           if len(s.split()) >= 4 and not s.endswith('?')]
        
        if not factual_sentences:
            scores.append(1.0)  # No substantive claims to check
            continue
        
        # Build prompt focused on finding contradictions
        prompt = """You are evaluating whether statements from an AI response CONTRADICT information in a reference context.
        Your ONLY job is to identify direct contradictions - statements that directly oppose facts stated in the reference.

        Reference Context:
        ```
        {context}
        ```

        Response statements to check:
        {statements}

        For each statement, determine if it CONTRADICTS the reference context.
        - Label as "CONTRADICTION" only if the statement directly opposes a fact in the reference.
        - Label as "NO_CONTRADICTION" if the statement is supported by or simply not mentioned in the reference.

        IMPORTANT: Additional information not found in the reference is fine as long as it doesn't contradict it.
        Only flag direct factual contradictions, not opinions, style differences, or added details.

        Format your response as a JSON object with a "labels" key containing an array of the labels, like this:
        {{"labels": ["NO_CONTRADICTION", "CONTRADICTION", ...]}}
        """
        
        # Process sentences in batches for efficiency
        batch_size = 5
        contradiction_count = 0
        total_sentences = len(factual_sentences)
        
        for i in range(0, total_sentences, batch_size):
            batch = factual_sentences[i:i+batch_size]
            
            # Format statements for the prompt
            statements_text = "\n".join([f"{j+1}. {s}" for j, s in enumerate(batch)])
            
            # Send request to check for contradictions
            messages = [{"role": "user", "content": prompt.format(context=context, statements=statements_text)}]
            
            try:
                response_eval = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use a smaller, faster model for efficiency
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                # Parse the response
                result = response_eval.choices[0].message.content
                labels = json.loads(result).get("labels", [])
                
                # Count contradictions
                contradiction_count += labels.count("CONTRADICTION")
                
            except Exception as e:
                print(f"Error checking contradictions: {e}")
                # Neutral stance on error
                contradiction_count += 0
        
        # Calculate faithfulness score based on absence of contradictions
        # 1.0 means no contradictions, lower scores indicate more contradictions
        if total_sentences > 0:
            # Calculate proportion of non-contradicting sentences
            faithfulness_score = 1.0 - (contradiction_count / total_sentences)
            scores.append(round(faithfulness_score, 3))
        else:
            scores.append(1.0)  # No sentences to check
            
    return scores

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
    # print(llm_grading(queries, ground_truths, model_answers, rubrics))
    
    reference = """"[{\"airbnb\": {\"Construction year\": 2019, \"NAME\": \"Only 2 stops to Manhattan studio\", \"availability 365\": 188.0, \"calculated host listings count\": 1, \"cancellation_policy\": \"moderate\", \"country\": \"United States\", \"country code\": \"US\", \"host id\": 80873428617, \"host name\": \"Allen & Irina\", \"host_identity_verified\": \"unconfirmed\", \"house_rules\": \"\", \"id\": \"30093739\", \"instant_bookable\": false, \"last review\": \"2022-02-20 00:00:00\", \"lat\": 40.70935, \"license\": \"\", \"long\": -73.95342, \"minimum nights\": 30, \"neighbourhood\": \"Williamsburg\", \"neighbourhood group\": \"Brooklyn\", \"number of reviews\": 184, \"price\": 468, \"review rate number\": 5, \"reviews per month\": 1.18, \"room type\": \"Entire home/apt\", \"service fee\": 94}}"
    """

    model_answer = """
    The construction year for Only 2 stops to manhattan studio is 2021. It is located in the neighborhood of brooklyn
    """
    
    print(hybrid_faithfulness(retrieved_contexts=[reference], responses=[model_answer]))
    print(contradiction_faithfulness(retrieved_contexts=[reference], responses=[model_answer]))