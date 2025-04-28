# src/data/generator.py
from openai import OpenAI
from typing import List, Dict, Any, Optional
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
from src.data.generator_v2 import DataGenerator

load_dotenv()

def init_generator(type="single-hop"):
    if type == "single-hop":
        return SyntheticDataGenerator()
    elif type == "multi-hop":
        return DataGenerator()
    else:
        raise ValueError(f"Invalid generator type: {type}. Options: single-hop, multi-hop")

class SyntheticDataGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.openai = OpenAI(api_key=self.api_key)

    def generate_questions(self, documents: List[str], metadata: str = None, **kwargs) -> List[str]:
        """
        Generate questions for the provided documents.
        Maintains conversation history for the previous 10 question generations.
        
        Args:
            documents: List of documents to generate questions for
            metadata: Optional metadata about the documents
            **kwargs: Additional parameters including:
                - custom_instructions: Optional custom instructions for question generation
                - example_questions: Optional list of example questions
        """
        metadata = metadata if metadata is not None else "Document is plain text, hence no metadata is provided."
        questions = []
        conversation_history = []
        
        rendered_prompt = render_synthetic_query_prompt(
            custom_instructions=kwargs.get("custom_instructions"),
            example_questions=kwargs.get("example_questions")
        )
        
        for document in tqdm(documents, desc="Generating questions"):
            messages = [
                {
                    "role": "developer",
                    "content": rendered_prompt
                }
            ]
            messages.extend(conversation_history)
            messages.append({
                "role": "user",
                "content": f"Document: {document}\n\nMetadata: {metadata}"
            })
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = messages,
            )            
            response = completion.choices[0].message
            questions.append(response.content) 
            conversation_history.append({
                "role": "user",
                "content": f"Document: {document}\n\nMetadata: {metadata}"
            })
            conversation_history.append({
                "role": "assistant",
                "content": response.content
            })            
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
        return questions

    def generate_answers(self, documents: List[str], questions: List[str], metadata: str = None, **kwargs) -> List[str]:
        """
        Generate answers with LLM
        
        Args:
            documents: List of documents
            questions: List of questions
            metadata: Optional metadata about the documents
            **kwargs: Additional parameters including:
                - answer_style: Optional style instructions for the answers
                - answer_format: Optional format instructions for the answers
                - tone: Optional tone for the answers
                - max_length: Optional maximum length for answers
                - include_citations: Whether to include citations
                - additional_instructions: Optional additional instructions
                - custom_instructions: Optional custom instructions that override default behavior
        """
        answers = []
        metadata = metadata if metadata is not None else "Document is plain text, hence no metadata is provided."
        
        rendered_prompt = render_synthetic_valid_answer_prompt(
            answer_style=kwargs.get("answer_style"),
            answer_format=kwargs.get("answer_format"),
            tone=kwargs.get("tone"),
            max_length=kwargs.get("max_length"),
            include_citations=kwargs.get("include_citations", False),
            additional_instructions=kwargs.get("additional_instructions"),
            custom_instructions=kwargs.get("custom_instructions")
        )
        
        for question, document in tqdm(zip(questions, documents), desc="Generating answers", total=len(questions)):
            if question == "NO_QUESTION_POSSIBLE":
                answers.append("NO_ANSWER_POSSIBLE")
                continue
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role": "system",
                        "content": rendered_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Document: {document}\n\nMetadata: {metadata}\n\nQuestion: {question}"
                    }
                ]
            )
            candidate_answer = completion.choices[0].message.content
            
            candidate_answer = candidate_answer.replace("```json", "").replace("```", "")
            
            candidate_answer = json.loads(candidate_answer)
            
            answers.append(candidate_answer["answers"])
            
        return answers
    
    def synthesize_from_text(self, documents: List[str], metadata=None, expand=False, output_path: str = "generation.json", **kwargs):
        """
        Synthesize questions and answers for the provided documents.
        
        Args:
            documents: List of documents
            metadata: Optional metadata about the documents
            expand: Whether to expand documents
            output_path: Path to save the generated data
            **kwargs: Additional parameters for question and answer generation including:
                - question_custom_instructions: Custom instructions for question generation
                - example_questions: Example questions for question generation
                - answer_style: Style instructions for the answers
                - answer_format: Format instructions for the answers
                - tone: Tone for the answers
                - max_length: Maximum length for answers
                - include_citations: Whether to include citations
                - additional_instructions: Additional instructions for answer generation
                - answer_custom_instructions: Custom instructions for answer generation
            
        Returns:
            questions: List[str] -> List of questions
            answers: List[List[str]] -> List of list of answers
            reference_contexts: List[str] -> List of reference contexts
        """
        reference_contexts = documents
        
        print("Starting synthesis...")
        questions = self.generate_questions(
            documents, 
            metadata, 
            custom_instructions=kwargs.get("question_custom_instructions"),
            example_questions=kwargs.get("example_questions")
        )
        
        answers = self.generate_answers(
            documents, 
            questions, 
            metadata,
            answer_style=kwargs.get("answer_style"),
            answer_format=kwargs.get("answer_format"),
            tone=kwargs.get("tone"),
            max_length=kwargs.get("max_length"),
            include_citations=kwargs.get("include_citations", False),
            additional_instructions=kwargs.get("additional_instructions"),
            custom_instructions=kwargs.get("answer_custom_instructions")
        )        
                
        valid_indices = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if question != "NO_QUESTION_POSSIBLE" or answer != "NO_ANSWER_POSSIBLE":
                valid_indices.append(i)
        
        filtered_questions = [questions[i] for i in valid_indices]
        filtered_answers = [answers[i] for i in valid_indices]
        filtered_contexts = [reference_contexts[i] for i in valid_indices]
        
        generation = []
        for question, answer, reference in zip(filtered_questions, filtered_answers, filtered_contexts):
            generation.append({"question": question, "answers": answer, "reference": reference})
            
        with open(output_path, "w") as f:
            json.dump(generation, f, indent=4)  
        
        return {
            "questions": filtered_questions,
            "answers": filtered_answers,
            "reference_contexts": filtered_contexts
        }
        
    def synthesize_from_csv(self, path: str, field: str = None, metadata: str = None, output_path: str = "generation.json", limit: int = None, **kwargs):
        """
        Synthesize questions and answers for the provided csv file.
        
        Args:
            path: Path to the CSV file
            field: Field name in the CSV to use
            metadata: Optional metadata about the documents
            output_path: Path to save the generated data
            limit: Maximum number of records to process
            **kwargs: Additional parameters for question and answer generation
        """
        df = pd.read_csv(path)
        df = df.fillna("null")
        
        if limit is not None:
            df = df[:limit]
            
        df.to_json("data.json", orient="records")
        documents = self.load_from_json("data.json", field)
        
        os.remove("data.json")
        
        return self.synthesize_from_text(
            documents, 
            metadata,
            output_path=output_path,
            **kwargs
        )
    
    def synthesize_from_json(self, path: str, field: str = None, metadata: str = None, output_path: str = "generation.json", limit: int = None, **kwargs):
        """
        Synthesize questions and answers directly from a JSON file or directory.
        
        Args:
            path: Path to the JSON file or directory containing JSON files
            field: Field name in the JSON object containing the raw documents
            metadata: Optional metadata about the documents
            output_path: Path to save the generated data
            limit: Maximum number of records to process
            **kwargs: Additional parameters for question and answer generation
        """
        documents = self.load_from_json(path, field)
        
        if limit is not None:
            documents = documents[:limit]
        
        return self.synthesize_from_text(
            documents, 
            metadata,
            output_path=output_path,
            **kwargs
        )
        
    def load_from_json(self, path: str, field: str = None):
        """
        Loads the raw documents from a json file.
        
        Input arguments:
        path: str -> Path to the folder containing the json files or path to the file containing a single json file with multiple json objects.
        field: str -> Field name in the json object containing the raw documents.
        
        Returns a list of raw documents.
        """
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
    
    
    
if __name__ == "__main__":
    import time
    import psutil
    import os
    import argparse
    import tiktoken
    
    parser = argparse.ArgumentParser(description='Generate synthetic data from CSV or JSON files')
    parser.add_argument('--path', type=str, required=True, help='Path to the CSV or JSON file')
    parser.add_argument('--metadata-file', type=str, required=True, help='Path to a .txt file containing metadata description')
    parser.add_argument('--field', type=str, help='Field name in JSON to use (optional)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows to process (optional)')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], required=True, help='File format (csv or json)')
    
    parser.add_argument('--question-instructions', type=str, help='Custom instructions for question generation')
    parser.add_argument('--example-questions', type=str, help='Comma-separated list of example questions')
    
    parser.add_argument('--answer-style', type=str, help='Style instructions for answers')
    parser.add_argument('--answer-format', type=str, help='Format instructions for answers')
    parser.add_argument('--tone', type=str, help='Tone for answers')
    parser.add_argument('--max-length', type=int, help='Maximum length for answers')
    parser.add_argument('--include-citations', action='store_true', help='Include citations in answers')
    parser.add_argument('--additional-instructions', type=str, help='Additional instructions for answer generation')
    parser.add_argument('--answer-instructions', type=str, help='Custom instructions for answer generation')
    
    args = parser.parse_args()
    
    with open(args.metadata_file, 'r') as f:
        metadata = f.read().strip()
    
    example_questions = None
    if args.example_questions:
        example_questions = [q.strip() for q in args.example_questions.split(',')]
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024
    
    generator = SyntheticDataGenerator()
    
    if args.format == 'csv':
        if args.limit:
            df = pd.read_csv(args.path)
            df = df[:args.limit]
            temp_path = "temp_cleaned_data.csv"
            df.to_csv(temp_path, index=False)
            generated_data = generator.synthesize_from_csv(
                path=temp_path, 
                metadata=metadata,
                question_custom_instructions=args.question_instructions,
                example_questions=example_questions,
                answer_style=args.answer_style,
                answer_format=args.answer_format,
                tone=args.tone,
                max_length=args.max_length,
                include_citations=args.include_citations,
                additional_instructions=args.additional_instructions,
                answer_custom_instructions=args.answer_instructions
            )
            os.remove(temp_path)
        else:
            generated_data = generator.synthesize_from_csv(
                path=args.path, 
                metadata=metadata,
                question_custom_instructions=args.question_instructions,
                example_questions=example_questions,
                answer_style=args.answer_style,
                answer_format=args.answer_format,
                tone=args.tone,
                max_length=args.max_length,
                include_citations=args.include_citations,
                additional_instructions=args.additional_instructions,
                answer_custom_instructions=args.answer_instructions
            )
    else:
        if args.limit:
            documents = generator.load_from_json(path=args.path, field=args.field)
            documents = documents[:args.limit]
            temp_path = "temp_limited_data.json"
            with open(temp_path, 'w') as f:
                json.dump(documents, f)
            generated_data = generator.synthesize_from_json(
                path=temp_path,
                metadata=metadata,
                question_custom_instructions=args.question_instructions,
                example_questions=example_questions,
                answer_style=args.answer_style,
                answer_format=args.answer_format,
                tone=args.tone,
                max_length=args.max_length,
                include_citations=args.include_citations,
                additional_instructions=args.additional_instructions,
                answer_custom_instructions=args.answer_instructions
            )
            os.remove(temp_path)
        else:
            generated_data = generator.synthesize_from_json(
                path=args.path,
                field=args.field,
                metadata=metadata,
                question_custom_instructions=args.question_instructions,
                example_questions=example_questions,
                answer_style=args.answer_style,
                answer_format=args.answer_format,
                tone=args.tone,
                max_length=args.max_length,
                include_citations=args.include_citations,
                additional_instructions=args.additional_instructions,
                answer_custom_instructions=args.answer_instructions
            )
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    
    questions = generated_data["questions"]
    answers = generated_data["answers"]
    reference_contexts = generated_data["reference_contexts"]
    
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for question, answer, reference_context in zip(questions, answers, reference_contexts):
        print("Question:", question)
        print("Answer:", answer)
        print("Reference context:", reference_context)
        print("\n\n")
        
        question_tokens = len(encoding.encode(question))
        answer_tokens = len(encoding.encode(str(answer)))
        total_tokens += question_tokens + answer_tokens
    
    processing_time = end_time - start_time
    tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
    seconds_per_token = processing_time / total_tokens if total_tokens > 0 else 0
    
    from tabulate import tabulate
    
    metrics_data = [
        ["Processing time", f"{processing_time:.2f} seconds"],
        ["Memory usage", f"{end_memory - start_memory:.2f} MB"],
        ["Generated question-answer pairs", f"{len(questions)}"],
        ["Total tokens generated", f"{total_tokens}"],
        ["Average tokens per second", f"{tokens_per_second:.2f}"],
        ["Average time per token", f"{seconds_per_token:.4f} seconds"]
    ]
    
    print("\n" + "="*50)
    print("Performance Metrics")
    print("="*50)
    print(tabulate(metrics_data, headers=["Metric", "Value"], numalign="left"))
    print("="*50 + "\n")
    
    with open("generation.json", "w") as f:
        json.dump({"questions": questions, "answers": answers, "reference_contexts": reference_contexts}, f, indent=4)