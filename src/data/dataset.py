# src/data/dataset.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json

class EvalDataset(BaseModel):
    """
    Ground truth evaluation dataset
    """
    questions: Optional[List[str]] = Field(
        default = None,
        description="List of questions",
        validate_default=True
    )
    answers: Optional[List[str]] = Field(
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
    retrieved_contexts: Optional[List[str]] = Field(
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
                "answer": self.answers[i] if self.answers and i < len(self.answers) else None,
                "response": self.responses[i] if self.responses and i < len(self.responses) else None,
                "reference_context": self.reference_contexts[i] if self.reference_contexts and i < len(self.reference_contexts) else None,
                "retrieved_context": self.retrieved_contexts[i] if self.retrieved_contexts and i < len(self.retrieved_contexts) else None
            }
            for i in range(max_len)
        ]
        # Save to json file
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4) 

        return data
    
    # Checks that all input lists are of the same length
    @field_validator("questions", "answers", "responses", "reference_contexts", "retrieved_contexts", mode="before")
    @classmethod
    def validate_length(cls, v, info):
        data = info.data  

        # Filter out None fields
        filtered_data = {field: data[field] for field in {"questions", "answers", "responses", "reference_contexts", "retrieved_contexts"} if field in data and data[field] is not None}

        # If at least one field is provided, check lengths
        if filtered_data:
            lengths = [len(value) for value in filtered_data.values()]
            if len(set(lengths)) != 1:
                raise ValueError("All input lists must be of the same length")
        return v
    
    
    
if __name__ == '__main__':
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": ["Paris", "Joe Biden"],
        "responses": ["Paris", "Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"],
        "retrieved_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"]
    }
    dataset = EvalDataset(**data)
    print(dataset)
    print(dataset.to_json())