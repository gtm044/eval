# src/data/generator.py
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import json
import os
import pandas as pd
from tqdm import tqdm
from src.utils.prompts import (
    synthetic_query_prompt, 
    synthetic_valid_answer_prompt, 
    expand_documents_prompt,
    render_synthetic_query_prompt,
    render_synthetic_valid_answer_prompt,
    render_expand_documents_prompt
)

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
            "answers": [[filtered_answer] for filtered_answer in filtered_answers],
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
    import argparse
    import tiktoken
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic data from CSV or JSON files')
    parser.add_argument('--path', type=str, required=True, help='Path to the CSV or JSON file')
    parser.add_argument('--metadata-file', type=str, required=True, help='Path to a .txt file containing metadata description')
    parser.add_argument('--field', type=str, help='Field name in JSON to use (optional)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows to process (optional)')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], required=True, help='File format (csv or json)')
    args = parser.parse_args()
    
    # Read metadata from the provided file
    with open(args.metadata_file, 'r') as f:
        metadata = f.read().strip()
    
    # Start measuring time and memory
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    
    generator = SyntheticDataGenerator()
    
    if args.format == 'csv':
        # Load and preprocess CSV data if needed
        if args.limit:
            df = pd.read_csv(args.path)
            df = df[:args.limit]
            temp_path = "temp_cleaned_data.csv"
            df.to_csv(temp_path, index=False)
            generated_data = generator.synthesize_from_csv(path=temp_path, metadata=metadata)
            os.remove(temp_path)
        else:
            generated_data = generator.synthesize_from_csv(path=args.path, metadata=metadata)
    else:  # json
        documents = generator.load_from_json(path=args.path, field=args.field)
        if args.limit:
            documents = documents[:args.limit]
        generated_data = generator.synthesize(documents=documents, metadata=metadata)
    
    # End measuring time and memory
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB
    
    questions = generated_data["questions"]
    answers = generated_data["answers"]
    reference_contexts = generated_data["reference_contexts"]
    
    # Calculate token counts using tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # Using GPT-4o model encoding
    total_tokens = 0
    
    for question, answer, reference_context in zip(questions, answers, reference_contexts):
        print("Question:", question)
        print("Answer:", answer)
        print("Reference context:", reference_context)
        print("\n\n")
        
        # Count tokens for each item
        question_tokens = len(encoding.encode(question))
        answer_tokens = len(encoding.encode(str(answer)))
        total_tokens += question_tokens + answer_tokens
    
    # Calculate token generation metrics
    processing_time = end_time - start_time
    tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
    seconds_per_token = processing_time / total_tokens if total_tokens > 0 else 0
    
    # Print performance metrics
    from tabulate import tabulate
    
    metrics_data = [
        ["Processing time", f"{processing_time:.2f} seconds"],
        ["Memory usage", f"{end_memory - start_memory:.2f} MB"],
        ["Generated question-answer pairs", f"{len(questions)}"],
        ["Total tokens generated", f"{total_tokens}"],
        ["Average tokens per second", f"{tokens_per_second:.2f}"],
        ["Average time per token", f"{seconds_per_token:.4f} seconds"]
    ]
    
    # Print the table
    print("\n" + "="*50)
    print("Performance Metrics")
    print("="*50)
    print(tabulate(metrics_data, headers=["Metric", "Value"], numalign="left"))
    print("="*50 + "\n")
    
    # Save the questions and answers to a json file
    with open("generation.json", "w") as f:
        json.dump({"questions": questions, "answers": answers, "reference_contexts": reference_contexts}, f, indent=4)