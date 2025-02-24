# src/data/generator.py
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
import json
import os

load_dotenv()

synthetic_query_prompt: str = (
    "You are an expert question-answering system provided with a document. You must create a question for the provided document. "
    "The question must be answerable within the context of the document.\n\n"
    "The response should contain only the question without any additional text or formatting.\n\n"
)
synthetic_valid_answer_prompt: str = (
    "You are an expert question-answering system provided with a <question and a <document>. You must create an answer for the provided question. "
    "The answer must be answerable within the context of the document. "
    "Provide the answer alone without any additional text or formatting.\n\n"
)
expand_documents_prompt: str = (
    "You are an expert in document expansion. You are provided with a document that needs to be expanded. \n\n"
    "This is done to create a bigger dataset given a small document to test a RAG system.\n\n"
    "Expanding a document involves generating multiple variations of a given document with different wording, sentence structure, and phrasing while ensuring the core content remains intact. The output should maintain coherence, factual accuracy, and natural readability. \n\n"
    "The response should contain only the expanded document according to the json schema given below without any additional text, formatting or whitespaces.\n\n"
    "The output should be a json string containing a dictionary with the list of expanded documents, dont include the input document in the response.\n\n"
    "The number of documents to generate is not fixed and is decided by the content of the document, if you reach a point where you can't paraphrase anymore to create more meaningful documents, stop.\n\n"
    "But if there is a number specified in an input field <limit>, you should generate that many documents.\n\n"
)

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
        for question, document in zip(questions, documents):
            completion = self.openai.chat.completions.create(
                model = "gpt-4o",
                messages = [
                    {
                        "role": "developer",
                        "content": synthetic_valid_answer_prompt
                    },
                    {
                        "role": "user",
                        "content": "Document: "+document+"\n\nQuestion: "+question
                    }
                ]
            )
            response = completion.choices[0].message
            answers.append(response.content)
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
    
if __name__ == '__main__':
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Paris was invented in 1984",
        "Bob the builder was busy on a sunny day."
    ]
    
    generator = SyntheticDataGenerator()
    
    # Expand your raw documents
    expanded_documents = generator.expand(documents)
    
    # Synthesize the ground truth (questions, ground truth answers, reference contexts)
    print(generator.synthesize(expanded_documents))