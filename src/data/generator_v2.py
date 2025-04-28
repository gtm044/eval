# A cleaner, multi-hop implementation of `generator.py`.
# Will deprecate `generator.py` in the future, after integrating single-hop generation and custom instructions.

from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import os
import pandas as pd
from tqdm import tqdm
from src.utils.prompts import (
    render_complex_query_prompt,
    complex_query_prompt,
    render_complex_answer_prompt,
    complex_answer_prompt
)
from src.data.cluster import SemanticCluster
from src.data.cluster.utils import get_default_save_directory


class DataGenerator:
    def __init__(self, api_key=None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        
    def generate_query(self, cluster_json: str):
        metadata = "No metadata is provided."
        messages = [
            {"role": "developer", "content": complex_query_prompt},
            {"role": "user", "content": f"Document Cluster: {cluster_json}\n\nMetadata: {metadata}"}
        ]
        max_retries = 5
        for _ in range(max_retries):
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            response = completion.choices[0].message.content
            
            if not (response.startswith('{') or response.startswith('```')):
                return response
                
            try:
                if response.startswith('```'):
                    response = response.strip('```json\n').strip('\n```')
                if response.startswith('{'):
                    parsed = json.loads(response)
                    if 'question' in parsed:
                        return parsed['question']
                    elif 'questions' in parsed and isinstance(parsed['questions'], list) and len(parsed['questions']) > 0:
                        return parsed['questions'][0]
            except:
                pass
            messages.append({"role": "user", "content": "Please provide just the question text without any JSON formatting or code blocks."})
        return "NO_QUESTION_POSSIBLE"


    def generate_answer(self, cluster_json: str, question: str, **kwargs):
        """
        kwargs:
            answer_style: str
            answer_format: str
            tone: str
            max_length: int
            include_citations: bool
            additional_instructions: str
            custom_instructions: str
        """
        rendered_prompt = render_complex_answer_prompt(
            answer_style=kwargs.get("answer_style", None),
            answer_format=kwargs.get("answer_format", None),
            tone=kwargs.get("tone", None),
            max_length=kwargs.get("max_length", None),
            include_citations=kwargs.get("include_citations", None),
            additional_instructions=kwargs.get("additional_instructions", None),
            custom_instructions=kwargs.get("custom_instructions", None)
        )
        
        metadata = "No metadata is provided."
        messages = [
            {"role": "developer", "content": rendered_prompt},
            {"role": "user", "content": f"Question: {question}\n\nDocument Cluster: {cluster_json}\n\nMetadata: {metadata}"}
        ]
                
        max_retries = 5
        for _ in range(max_retries):
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            response = completion.choices[0].message.content
            
            try:
                if response.startswith('```'):
                    response = response.strip('```json\n').strip('\n```')
                parsed_response = json.loads(response)
                
                if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                    if len(parsed_response["answers"]) == 3:
                        return parsed_response["answers"]
                    else:
                        messages.append({"role": "user", "content": "Please provide exactly 3 answers in the 'answers' array."})
                else:
                    messages.append({"role": "user", "content": "Your response must be a valid JSON with an 'answers' field containing an array of exactly 3 answers."})
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "Your response must be a valid JSON object with an 'answers' array containing exactly 3 answers."})
        
        return ["NO_ANSWER_POSSIBLE"] * 3


    def process_clusters(self, clusters, metadata=None, **kwargs):
        references = []
        questions = []
        
        for cluster in tqdm(clusters, desc="Generating questions"):
            reference = []
            for c in cluster:
                reference.append(json.dumps(c))
            references.append(reference)
            cluster_json = json.dumps(cluster)
            synthetic_queries = self.generate_query(cluster_json)
            questions.append(synthetic_queries)
        
        answers = []
        for cluster, question in tqdm(zip(clusters, questions), desc="Generating answers", total=len(questions)):
            if question != "NO_QUESTION_POSSIBLE":
                cluster_json = json.dumps(cluster)
                answer = self.generate_answer(cluster_json, question, **kwargs)
                answers.append(answer)
            else:
                answers.append("NO_ANSWER_POSSIBLE")
        
        generation = []
        for question, answer, reference in zip(questions, answers, references):
            generation.append({"question": question, "answer": answer, "reference": reference})
            
        return generation


    def save_results(self, generation, output_path="synthetic_data.json"):
        with open(output_path, "w") as f:
            json.dump(generation, f, indent=4)
        return output_path


    def process_from_clusters(self, metadata=None, **kwargs):
        cluster_path = get_default_save_directory()
        cluster_files = os.listdir(cluster_path)
        clusters = []
        
        for cluster_file in cluster_files:
            with open(os.path.join(cluster_path, cluster_file), "r") as f:
                cluster = json.load(f)
                clusters.append(cluster)
                
        for cluster_file in cluster_files:
            os.remove(os.path.join(cluster_path, cluster_file))
        os.rmdir(cluster_path)
        
        return self.process_clusters(clusters, metadata, **kwargs)


    def synthesize_from_json(self, json_path: str, field: str = None, limit: int = None, metadata=None, output_path="synthetic_data.json", **kwargs):
        """
        Synthesize questions and answers from a JSON file
        
        Args:
            json_path: Path to the JSON file
            field: Field name in JSON to use for document content
            limit: Maximum number of records to process
            metadata: Optional metadata about the documents
            output_path: Path to save the generated data
        """
        cluster_engine = SemanticCluster()
        cluster_engine.process_json(json_path, field=field, limit=limit)
        cluster_engine.build_clusters()
        generation = self.process_from_clusters(metadata, **kwargs)
        self.save_results(generation, output_path)
        return generation


    def synthesize_from_csv(self, csv_path: str, field: str = None, limit=None, metadata=None, output_path="synthetic_data.json", **kwargs):
        """
        Synthesize questions and answers from a CSV file
        
        Args:
            csv_path: Path to the CSV file
            field: Field name in CSV to use for document content
            limit: Maximum number of records to process
            metadata: Optional metadata about the documents
            output_path: Path to save the generated data
        """
        cluster_engine = SemanticCluster()
        cluster_engine.process_csv(csv_path, field=field, limit=limit)
        cluster_engine.build_clusters()
        generation = self.process_from_clusters(metadata, **kwargs)
        self.save_results(generation, output_path)
        return generation


    def synthesize_from_text(self, texts: List[str], metadata=None, output_path="synthetic_data.json", **kwargs):
        documents = [{"text": text} for text in texts]
        cluster_engine = SemanticCluster(texts=documents)
        cluster_engine.build_clusters()
        generation = self.process_from_clusters(metadata, **kwargs)
        self.save_results(generation, output_path)
        return generation


if __name__ == "__main__":
    generator = DataGenerator()
    csv_path = "/Users/goutham.krishnan/Documents/Work/eval/input_data/airbnb.csv"
    generator.synthesize_from_csv(csv_path, limit=50, output_path="airbnb_synthetic_data2.json", answer_style="detailed", answer_format="json", tone="formal", max_length=100, include_citations=False, additional_instructions="Use the references to support your answer.", custom_instructions="Ensure the answer is concise and to the point.")