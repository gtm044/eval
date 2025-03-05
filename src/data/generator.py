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
    
    def expand(self, documents: List[str], limit=5):
        """
        Expand the given document by paraphrasing it.
        """
        expanded_documents = []
        for document in tqdm(documents, desc="Expanding documents"):
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
        
    
    def synthesize(self, documents: List[str], metadata=None, expand=False):
        """
        Synthesize questions and answers for the provided documents.
        """
        reference_contexts = documents
        
        print("Starting synthesis process...")
        questions = self.generate_questions(documents, metadata)
        answers = self.generate_answers(documents, questions, metadata)        
        # Expand the dataset (logic here)
        
        return {
            "questions": questions,
            "answers": answers,
            "reference_contexts": reference_contexts
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
    
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Paris was invented in 1984",
        "Bob the builder was busy on a sunny day."
    ]
    
    # Load the data
    df = pd.read_csv("/Users/goutham.krishnan/Documents/Work/eval/src/data/data.csv")
    df = df[:10]
    
    # Drop the column "license"
    df = df.drop(columns=["license"])
    
    # Clean the data
    df = df.dropna()
    
    # Save the dataframe to a csv file
    df.to_csv("cleaned_data.csv", index=False)
    
    # If generating from a .csv file, then metadata should be provided as a string
    
    metadata = """
    [{"Construction year": "number", "NAME": "string", "availability 365": ["number", "string"], "calculated host listings count": ["number", "string"], "cancellation_policy": "string", "country": "string", "country code": "string", "host id": "number", "host name": "string", "host_identity_verified": "string", "house_rules": "string", "id": "string", "instant_bookable": ["boolean", "string"], "last review": "string", "lat": "number", "license": "string", "long": "number", "minimum nights": ["number", "string"], "neighbourhood": "string", "neighbourhood group": "string", "number of reviews": ["number", "string"], "price": ["number", "string"], "review rate number": ["number", "string"], "reviews per month": ["number", "string"], "room type": "string", "service fee": ["number", "string"]}, {"Construction year": "Year when the building was constructed", "NAME": "Name of the Airbnb listing", "availability 365": "The availability of the listing x days in the future as determined by the calendar.", "calculated host listings count": "Number of properties listed by the host", "cancellation_policy": "Cancellation policy applied to this listing, strict , moderate , flexible", "country": "Country where the listing is located", "country code": "Two-letter country code (ISO 3166-1 alpha-2)", "host id": "Airbnb's unique identifier for the host", "host name": "Name of the host,usually only first name", "host_identity_verified": "Indicates whether the host's identity is verified", "house_rules": "Set of rules defined by the host for guests", "id": "Unique identifier for the listing", "instant_bookable": "[t=true; f=false]. Whether the guest can automatically book the listing without the host requiring to accept their", "last review": "Date of the last guest review, format: YYYY-MM-DD HH:MM:SS", "lat": "Latitude coordinate of the listing", "license": "License number required for short-term rental compliance", "long": "Longitude coordinate of the listing", "minimum nights": "Minimum number of nights required for booking", "neighbourhood": "Represents a specific local area or district within a city. Examples include Kensington and Harlem. It provides more granular location details.", "neighbourhood group": "Larger area grouping multiple neighborhoods . A broader classification that groups multiple neighbourhoods together. Examples include Brooklyn and Manhattan. This helps categorize properties into larger city regions.", "number of reviews": "Total number of reviews received by the listing", "price": "Cost per night in USD", "review rate number": "Overall rating given by guests", "reviews per month": "Average number of reviews per month", "room type": "Airbnb hosts can list entire homes/apartments, private, shared rooms, and more recently hotel rooms.Depending on type", "service fee": "Additional service charge applied per booking"}]
    """
    
    generator = SyntheticDataGenerator()
    genrated_data = generator.synthesize_from_csv(path="cleaned_data.csv", metadata="Document is a json object, hence metadata is provided.")
    
    questions = genrated_data["questions"]
    answers = genrated_data["answers"]
    reference_contexts = genrated_data["reference_contexts"]
    
    for question, answer, reference_context in zip(questions, answers, reference_contexts):
        print("Question:", question)
        print("Answer:", answer)
        print("Reference context:", reference_context)
        print("\n\n")