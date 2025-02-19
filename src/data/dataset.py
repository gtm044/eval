# src/data/dataset.py
from pydantic import BaseModel, Field, field_validator
from typing import List
import json

class EvalDataset(BaseModel):
    """
    Ground truth evaluation dataset
    """
    questions: List[str] = Field(
        description="List of questions"
    )
    answers: List[str] = Field(
        description="List of ground truth answers"
    )
    responses: List[str] = Field(
        description="List of model responses"
    )
    reference_contexts: List[str] = Field(
        description="List of reference contexts"
    )
    retrieved_contexts: List[str] = Field(
        description="List of contexts retrieved for the question by the vector store"
    )
    
    # Checks that all input lists are of the same length
    @field_validator("questions", "answers", "responses", "reference_contexts", "retrieved_contexts", mode="before")
    @classmethod
    def validate_length(cls, v, info):
        data = info.data  
        if data:
            lengths = [len(data[field]) for field in data if field in {"questions", "answers", "responses", "reference_contexts", "retrieved_contexts"}]
            if len(set(lengths)) != 1:
                raise ValueError("All input lists must be of the same length")
        return v
    
    # Convert the dataset into a list of json objects
    def to_json(self, filename=None):
        # Convert list of lists into a list of individual objects
        data = [
            {
                "question": self.questions[i],
                "answer": self.answers[i],
                "response": self.responses[i],
                "reference_context": self.reference_contexts[i],
                "retrieved_context": self.retrieved_contexts[i]
            }
            for i in range(len(self.questions))
        ]

        # Save to file
        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4) 

        return json.dumps(data, indent=4)
    
    
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