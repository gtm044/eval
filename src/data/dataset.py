# src/data/dataset.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
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
    # Agentic data
    agent_responses: Optional[List[List[str]]] = Field(
        default = None,
        description="List of final agent responses",
        validate_default=True
    )
    agent_tool_calls: Optional[List[List[Dict[str, Any]]]] = Field(
        default = None,
        description="List of agent tool calls",
        validate_default=True
    )
    agent_tool_outputs: Optional[List[List[str]]] = Field(
        default = None,
        description="List of agent tool outputs",
        validate_default=True
    )
    # Agnentic ground truth data
    reference_tool_calls: Optional[List[List[Dict[str, Any]]]] = Field(
        default = None,
        description="List of reference tool calls",
        validate_default=True
    )
    gt_answers: Optional[List[List[str]]] = Field(
        default = None,
        description="List of ground truth answers",
        validate_default=True
    )
    gt_tool_outputs: Optional[List[List[str]]] = Field(
        default = None,
        description="List of ground truth tool outputs",
        validate_default=True
    )
        
    # Convert the dataset into a list of json objects
    def to_json(self, filename=None):
        # Get the maximum length of provided lists (handling None fields)
        max_len = max(len(lst) if lst is not None else 0 for lst in 
                    [self.questions, self.answers, self.responses, self.reference_contexts, self.retrieved_contexts, self.agent_responses, self.agent_tool_calls, self.agent_tool_outputs, self.reference_tool_calls, self.gt_answers, self.gt_tool_outputs])
        # Convert list of lists into a list of individual objects
        data = [
            {
                "question": self.questions[i] if self.questions and i < len(self.questions) else None,
                "answers": self.answers[i] if self.answers and i < len(self.answers) else None,
                "response": self.responses[i] if self.responses and i < len(self.responses) else None,
                "reference_context": self.reference_contexts[i] if self.reference_contexts and i < len(self.reference_contexts) else None,
                "retrieved_contexts": self.retrieved_contexts[i] if self.retrieved_contexts and i < len(self.retrieved_contexts) else None,
                "agent_responses": self.agent_responses[i] if self.agent_responses and i < len(self.agent_responses) else None,
                "agent_tool_calls": self.agent_tool_calls[i] if self.agent_tool_calls and i < len(self.agent_tool_calls) else None,
                "agent_tool_outputs": self.agent_tool_outputs[i] if self.agent_tool_outputs and i < len(self.agent_tool_outputs) else None,
                "reference_tool_calls": self.reference_tool_calls[i] if self.reference_tool_calls and i < len(self.reference_tool_calls) else None,
                "gt_answers": self.gt_answers[i] if self.gt_answers and i < len(self.gt_answers) else None,
                "gt_tool_outputs": self.gt_tool_outputs[i] if self.gt_tool_outputs and i < len(self.gt_tool_outputs) else None,
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
            
        questions = []
        answers = []
        responses = []
        reference_contexts = []
        retrieved_contexts = []
        agent_responses = []
        agent_tool_calls = []
        agent_tool_outputs = []
        reference_tool_calls = []
        gt_answers = []
        gt_tool_outputs = []
        
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
            if doc.get("agent_responses") is not None:
                agent_responses.append(doc["agent_responses"])
            if doc.get("agent_tool_calls") is not None:
                agent_tool_calls.append(doc["agent_tool_calls"])
            if doc.get("agent_tool_outputs") is not None:
                agent_tool_outputs.append(doc["agent_tool_outputs"])
            if doc.get("reference_tool_calls") is not None:
                reference_tool_calls.append(doc["reference_tool_calls"])
            if doc.get("gt_answers") is not None:
                gt_answers.append(doc["gt_answers"])
            if doc.get("gt_tool_outputs") is not None:
                gt_tool_outputs.append(doc["gt_tool_outputs"])
        
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
        if agent_responses:
            dataset_dict["agent_responses"] = agent_responses
        if agent_tool_calls:
            dataset_dict["agent_tool_calls"] = agent_tool_calls
        if agent_tool_outputs:
            dataset_dict["agent_tool_outputs"] = agent_tool_outputs
        if reference_tool_calls:
            dataset_dict["reference_tool_calls"] = reference_tool_calls
        if gt_answers:
            dataset_dict["gt_answers"] = gt_answers
        if gt_tool_outputs:
            dataset_dict["gt_tool_outputs"] = gt_tool_outputs
            
        return cls(**dataset_dict)
    
    # Checks that all input lists are of the same length (for the RAG part)
    @field_validator("questions", "answers", "responses", "reference_contexts", "retrieved_contexts", mode="before")
    @classmethod
    def validate_length(cls, v, info):
        data = info.data  
        filtered_data = {field: data[field] for field in {"questions", "answers", "responses", "reference_contexts", "retrieved_contexts"} if field in data and data[field] is not None}
        error_message = ""
        for field, value in filtered_data.items():
            error_message += f"{field}: {len(value)}\n"
        if filtered_data:
            lengths = []
            for field, value in filtered_data.items():
                lengths.append(len(value))
                    
            if len(set(lengths)) != 1:
                raise ValueError("All input lists must be of the same length\n" + error_message)
        return v
    
    
if __name__ == '__main__':
    
    ## RAG/General AI pipeline example
    # data = {
    #     "questions": ["What is the capital of France?", "Who is the president of the USA?"],
    #     "answers": [["Paris", "France"], ["Washington", "USA"]],
    #     "responses": ["Paris", "Joe Biden"],
    #     "reference_contexts": ["Paris is the capital of France", "The USA is a country in North America"],
    #     "retrieved_contexts": [["Paris is the capital of France", "France is a country in Europe"], ["Washington is the capital of the USA", "The USA is a country in North America"]]
    # }
    # dataset = EvalDataset(**data)
    # print(dataset)
    # print(dataset.to_json(filename="test.json"))
    
    ## Agentic example
    data = {
        "questions": ["What is the price of copper?", "What is the price of gold?"],
        "agent_responses": [["The current price of copper is $0.0098 per gram."], ["The current price of gold is $88.16 per gram."]],
        "agent_tool_calls": [
            [{"name": "get_price", "args": {"item": "copper"}}],
            [{"name": "get_price", "args": {"item": "gold"}}]
        ],
        "agent_tool_outputs": [["$0.0098"], ["$88.16"]],
        "reference_tool_calls": [
            [{"name": "get_price", "args": {"item": "copper"}}],
            [{"name": "get_price", "args": {"item": "gold"}}]
        ],
        "gt_answers": [["$0.0098 per gram"], ["$88.16 per gram"]],
        "gt_tool_outputs": [["$0.0098"], ["$88.16"]]
    }
    dataset = EvalDataset(**data)
    print(dataset)
    print(dataset.to_json(filename="test_agentic.json"))