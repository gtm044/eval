# src/data/dataset.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json
import uuid

class EvalDataset(BaseModel):
    """
    Ground truth evaluation dataset
    """
    dataset_id: Optional[str] = Field(
        default = str(uuid.uuid4()),
        description="Unique identifier for the evaluation dataset"
    )
    questions: Optional[List[str]] = Field(
        default = None,
        description="List of questions",
        validate_default=True
    )
    answers: Optional[List[List[str]]] = Field(
        default = None,
        description="List of ground truth answers",
        validate_default=True
    )
    responses: Optional[List[str]] = Field(
        default = None,
        description="List of model responses",
        validate_default=True
    )
    reference_contexts: Optional[List[str]] = Field(
        default = None,
        description="List of reference contexts",
        validate_default=True
    )
    retrieved_contexts: Optional[List[List[str]]] = Field(
        default = None,
        description="List of contexts retrieved for the question by the vector store",
        validate_default=True
    )
    
    # Convert the dataset into a list of json objects
    def to_json(self, filename=None):
        # Get the maximum length of provided lists (handling None fields)
        max_len = max(len(lst) if lst is not None else 0 for lst in 
                    [self.questions, self.answers, self.responses, self.reference_contexts, self.retrieved_contexts])
        # Convert list of lists into a list of individual objects
        data = [
            {
                "question": self.questions[i] if self.questions and i < len(self.questions) else None,
                "answers": self.answers[i] if self.answers and i < len(self.answers) else None,
                "response": self.responses[i] if self.responses and i < len(self.responses) else None,
                "reference_context": self.reference_contexts[i] if self.reference_contexts and i < len(self.reference_contexts) else None,
                "retrieved_contexts": self.retrieved_contexts[i] if self.retrieved_contexts and i < len(self.retrieved_contexts) else None
            }
            for i in range(max_len)
        ]
        # Save to json file
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4) 

        return data

    @classmethod
    def from_json(cls, json_path):
        """
        Create an EvalDataset from a list of JSON objects.
        
        Args:
            json_path: Path to the JSON file containing the data
                 
        Returns:
            EvalDataset: A new dataset instance
        """
        with open(json_path, "r") as f:
            data = json.load(f)
            
        # Initialize empty lists for each field
        questions = []
        answers = []
        responses = []
        reference_contexts = []
        retrieved_contexts = []
        
        # Extract data from each document
        for doc in data:
            if doc.get("question") is not None:
                questions.append(doc["question"])
            if doc.get("answers") is not None:
                answers.append(doc["answers"])
            if doc.get("response") is not None:
                responses.append(doc["response"])
            if doc.get("reference_context") is not None:
                reference_contexts.append(doc["reference_context"])
            if doc.get("retrieved_contexts") is not None:
                retrieved_contexts.append(doc["retrieved_contexts"])
        
        # Create dataset with non-empty fields
        dataset_dict = {}
        if questions:
            dataset_dict["questions"] = questions
        if answers:
            dataset_dict["answers"] = answers
        if responses:
            dataset_dict["responses"] = responses
        if reference_contexts:
            dataset_dict["reference_contexts"] = reference_contexts
        if retrieved_contexts:
            dataset_dict["retrieved_contexts"] = retrieved_contexts
            
        return cls(**dataset_dict)
    
    # Checks that all input lists are of the same length
    @field_validator("questions", "answers", "responses", "reference_contexts", "retrieved_contexts", mode="before")
    @classmethod
    def validate_length(cls, v, info):
        data = info.data  
        # Filter out None fields
        filtered_data = {field: data[field] for field in {"questions", "answers", "responses", "reference_contexts", "retrieved_contexts"} if field in data and data[field] is not None}
        # Log the length of each field in the filtered data
        error_message = ""
        for field, value in filtered_data.items():
            error_message += f"{field}: {len(value)}\n"
        # If at least one field is provided, check lengths
        if filtered_data:
            # Get the length of each field, accounting for answers and retrieved_contexts being lists of lists
            lengths = []
            for field, value in filtered_data.items():
                lengths.append(len(value))
                    
            if len(set(lengths)) != 1:
                raise ValueError("All input lists must be of the same length\n" + error_message)
        return v
    
    
if __name__ == '__main__':
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": [["Paris", "France"], ["Washington", "USA"]],
        "responses": ["Paris", "Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "The USA is a country in North America"],
        "retrieved_contexts": [["Paris is the capital of France", "France is a country in Europe"], ["Washington is the capital of the USA", "The USA is a country in North America"]]
    }
    dataset = EvalDataset(**data)
    print(dataset)
    print(dataset.to_json(filename="test.json"))