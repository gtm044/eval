# src/data/generator.py
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import json
import os
from src.utils.models import qa_bm25
from src.utils.prompts import synthetic_query_prompt, synthetic_valid_answer_prompt, expand_documents_prompt


load_dotenv()

class SyntheticDataGenerator:
    def __init__(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        # self.max_tokens = 5000
        self.openai = OpenAI(api_key=self.api_key)

    def generate_questions(self, documents: List[str]) -> List[str]:
        """
        Generate questions for the provided documents.
        """
        questions = []
        for document in documents:
            # prompt = synthetic_query_prompt.replace("<document>", document)
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role": "developer",
                        "content": synthetic_query_prompt
                    },
                    {
                        "role": "user",
                        "content": "Document: "+document
                    }
                ]
            )
            response = completion.choices[0].message
            questions.append(response.content)
        return questions

    def generate_answers(self, documents: List[str], questions: List[str]) -> List[str]:
        """
        Generate answers for the provided documents and questions.
        """
        answers = []
        # for question, document in zip(questions, documents):
        #     completion = self.openai.chat.completions.create(
        #         model = "gpt-4o",
        #         messages = [
        #             {
        #                 "role": "developer",
        #                 "content": synthetic_valid_answer_prompt
        #             },
        #             {
        #                 "role": "user",
        #                 "content": "Document: "+document+"\n\nQuestion: "+question
        #             }
        #         ]
        #     )
        #     response = completion.choices[0].message
        #     answers.append(response.content)

        for question,document in zip(questions, documents):
            answers.append(qa_bm25(document, question))
            
        return answers
    
    def expand(self, documents: List[str], limit=5):
        """
        Expand the given document by paraphrasing it.
        """
        expanded_documents = []
        for document in documents:
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role": "developer",
                        "content": expand_documents_prompt
                    },
                    {
                        "role": "user",
                        "content": "Input Document: "+document+"\n\nLimit: "+str(limit)
                    }
                ]
            )
            response = completion.choices[0].message
            for doc in (json.loads(response.content))["expanded_documents"]:
                expanded_documents.append(doc)
            
        return expanded_documents
        
    
    def synthesize(self, documents: List[str], expand=False):
        """
        Synthesize questions and answers for the provided documents.
        """
        reference_contexts = documents
        
        questions = self.generate_questions(documents)
        answers = self.generate_answers(documents, questions)        
        # Expand the dataset (logic here)
        
        return {
            "questions": questions,
            "answers": answers,
            "reference_contexts": reference_contexts
        }
        
    def load_from_json(self, path: str, field: str):
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
                    data = json.load(f)
                    documents.append(data[field])
        else:
            with open(path, "r") as f:
                data = json.load(f)
                documents = [doc[field] for doc in data]
                
        return documents
        
        
    
if __name__ == '__main__':
    
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Paris was invented in 1984",
        "Bob the builder was busy on a sunny day."
    ]
    
    generator = SyntheticDataGenerator()
    # documents = generator.load_from_json(path=os.environ["input_path"], field="reference_context")
    
    # Expand your raw documents
    # expanded_documents = generator.expand(documents, limit=3)
    
    # Synthesize the ground truth (questions, ground truth answers, reference contexts)
    print(generator.synthesize(documents))