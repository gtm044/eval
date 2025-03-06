# src/data/generator.py
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import json
import os
import pandas as pd
from tqdm import tqdm
from src.utils.models import qa_bm25
from src.utils.prompts import synthetic_query_prompt, synthetic_valid_answer_prompt, expand_documents_prompt

load_dotenv()

class SyntheticDataGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.openai = OpenAI(api_key=self.api_key)

    def generate_questions(self, documents: List[str], metadata: str = None) -> List[str]:
        """
        Generate questions for the provided documents.
        Maintains conversation history for the previous 10 question generations.
        """
        metadata = metadata if metadata is not None else "Document is plain text, hence no metadata is provided."
        questions = []
        conversation_history = []
        
        for document in tqdm(documents, desc="Generating questions"):
            # Build messages with conversation history
            messages = [
                {
                    "role": "developer",
                    "content": synthetic_query_prompt
                }
            ]
            # Add conversation history (up to 10 previous exchanges)
            messages.extend(conversation_history)
            # Add current document request
            messages.append({
                "role": "user",
                "content": f"Document: {document}\n\nMetadata: {metadata}"
            })
            # Generate question
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = messages,
            )            
            response = completion.choices[0].message
            questions.append(response.content) 
            # Update conversation history
            conversation_history.append({
                "role": "user",
                "content": f"Document: {document}\n\nMetadata: {metadata}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": response.content
            })            
            # Keep only the last 10 exchanges (20 messages - user and assistant pairs)
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
        return questions

    def generate_answers(self, documents: List[str], questions: List[str], metadata: str = None) -> List[str]:
        """
        Generate answers with LLM and verify with BM25
        """
        answers = []
        metadata = metadata if metadata is not None else "Document is plain text, hence no metadata is provided."
        
        for question, document in tqdm(zip(questions, documents), desc="Generating answers", total=len(questions)):
            # Generate candidate answer with LLM
            if question == "NO_QUESTION_POSSIBLE": # Dont generate answer for the tuple if question is "NO_QUESTION_POSSIBLE"
                answers.append("NO_ANSWER_POSSIBLE")
                continue
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role": "system",
                        "content": synthetic_valid_answer_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Document: {document}\n\nMetadata: {metadata}\n\nQuestion: {question}"
                    }
                ]
            )
            candidate_answer = completion.choices[0].message.content
            answers.append(candidate_answer)
        
        return answers
    
    def synthesize(self, documents: List[str], metadata=None, expand=False):
        """
        Synthesize questions and answers for the provided documents.
        Returns:
            questions: List[str] -> List of questions
            answers: List[List[str]] -> List of list of answers
            reference_contexts: List[str] -> List of reference contexts
        """
        reference_contexts = documents
        
        print("Starting synthesis...")
        questions = self.generate_questions(documents, metadata)
        answers = self.generate_answers(documents, questions, metadata)        
                
        # Filter out data points where question or answer is "NULL"
        valid_indices = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if question != "NO_QUESTION_POSSIBLE" or answer != "NO_ANSWER_POSSIBLE":
                valid_indices.append(i)
        
        filtered_questions = [questions[i] for i in valid_indices]
        filtered_answers = [answers[i] for i in valid_indices]
        filtered_contexts = [reference_contexts[i] for i in valid_indices]
        
        return {
            "questions": filtered_questions,
            "answers": [filtered_answers],
            "reference_contexts": filtered_contexts
        }
        
    def synthesize_from_csv(self, path: str, field: str = None, metadata: str = None):
        """
        Synthesize questions and answers for the provided csv file.
        """
        df = pd.read_csv(path)
        # Set all the null values to "null"
        df = df.fillna("null")
        df.to_json("data.json", orient="records")
        documents = self.load_from_json("data.json", field)
        
        # Delete the json file
        os.remove("data.json")
        
        return self.synthesize(documents, metadata)
        
        
    def load_from_json(self, path: str, field: str = None):
        """
        Loads the raw documents from a json file.
        
        Input arguments:
        path: str -> Path to the folder containing the json files or path to the file containing a single json file with multiple json objects.
        field: str -> Field name in the json object containing the raw documents.
        
        Returns a list of raw documents.
        """
        # Check if the given path is a directory or a file
        if os.path.isdir(path):
            files = os.listdir(path)
            documents = []
            for file in files:
                with open(os.path.join(path, file), "r") as f:
                    if field is not None:
                        data = json.load(f)
                        documents.append(data[field])
                    else:
                        documents.append(json.dumps(json.load(f)))
        else:
            with open(path, "r") as f:
                data = json.load(f)
                if field is not None:
                    documents = [doc[field] for doc in data]
                else:
                    documents = [json.dumps(doc) for doc in data]
                
        return documents

    
if __name__ == '__main__':
    import time
    import psutil
    import os
    
    ## Synthesizing ground truth from a .csv file
    PATH_TO_CSV = "/Users/goutham.krishnan/Documents/Work/eval/src/data/data.csv"
    
    # Start measuring time and memory
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    
    # Load the data
    df = pd.read_csv(PATH_TO_CSV)
    df = df[:10]
    
    # Simple preprocessing
    df = df.drop(columns=["license"])
    df = df.dropna()
    df.to_csv("cleaned_data.csv", index=False)
    
    # If generating from a .csv file, then metadata should be provided as a string
    metadata = """
    <provide metadata for the csv/json document>
    """
    
    generator = SyntheticDataGenerator()
    genrated_data = generator.synthesize_from_csv(path="cleaned_data.csv", metadata="Document is a json object, hence metadata is provided.")
    
    # End measuring time and memory
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    
    questions = genrated_data["questions"]
    answers = genrated_data["answers"]
    reference_contexts = genrated_data["reference_contexts"]
    
    for question, answer, reference_context in zip(questions, answers[0], reference_contexts):
        print("Question:", question)
        print("Answer:", answer)
        print("Reference context:", reference_context)
        print("\n\n")
    
    # Print performance metrics
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        

    ## Synthesizing ground truth from a json document(s)
    
    # # Load the data
    # documents = generator.load_from_json(path="<path to the json file>", field="description") # If there are multiple json docs, path should be the directory containing the json files
    
    # # Synthesize the data
    # genrated_data = generator.synthesize(documents, metadata)
    
    # questions = genrated_data["questions"]
    # answers = genrated_data["answers"]
    # reference_contexts = genrated_data["reference_contexts"]
    
    # for question, answer, reference_context in zip(questions, answers[0], reference_contexts):
    #     print("Question:", question)
    #     print("Answer:", answer)
    #     print("Reference context:", reference_context)
    #     print("\n\n")