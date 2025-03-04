# src/evaluator/validation.py
from src.evaluator.metrics import chunking, generation, retrieval
from src.data.dataset import EvalDataset
from datasets import Dataset
from typing import List, Optional
import json
import pandas as pd
import os
import ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness
# from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics.base import Metric

class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        metrics: Optional[List[Metric] | List[str]] = None,
        segments: Optional[List[str]] = None,
    ):
        """
        Initialize the ValidationEngine with dataset and validation options.
        
        Args:
            dataset: The dataset containing questions, answers, responses, and contexts
            metrics: List of metrics to evaluate
            segments: List of segments to evaluate
        """
        self.dataset = dataset
        self.metrics = metrics
        self.segments = segments
        
    
    def evaluate(self):
        """
        Calculate RAGAS metrics for the evaluation dataset.
        
        Returns:
            tuple: (results_dict, metrics_list, json_schema)
        """ 
        # Prepare dataset for single-turn evaluation
        golden_dataset = Dataset.from_dict(
            {
                "question": self.dataset.questions,
                "answer": self.dataset.responses,
                "contexts": self.dataset.retrieved_contexts,
                "ground_truths": self.dataset.answers,
                "reference": self.dataset.reference_contexts
            }
        )
            
        results = ragas.evaluate(dataset=golden_dataset, metrics=self.metrics)
        
        # Convert to pandas DataFrame
        df = results.to_pandas()
        
        # Create results directory if it doesn't exist
        results_dir = ".results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to CSV in the results directory
        csv_filename = os.path.join(results_dir, "results.csv")
        df.to_csv(csv_filename, index=False)
        
        # Convert to JSON and save in the results directory
        results_dict = df.to_dict(orient='records')
        json_filename = os.path.join(results_dir, "results.json")
        with open(json_filename, "w") as f:
            json.dump(results_dict, f, indent=4)
        
        # Create JSON schema
        json_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {}
            }
        }
        
        # Add properties to schema based on columns in the DataFrame
        for column in df.columns:
            json_schema["items"]["properties"][column] = {
                "type": "number" if df[column].dtype in ['float64', 'int64'] else "string",
                "description": f"RAGAS metric: {column}"
            }
        
        print(f"Results saved to {csv_filename} and {json_filename}")
        return results_dict, self.metrics, json_schema
        
        
if __name__=='__main__':
    # Example usage of ValidationEngine
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": [["Paris", "France"], ["Joe Biden", "USA"]],
        "responses": ["Capital of france is Paris", "President of the USA is Joe Biden"],
        "reference_contexts": ["Paris", "Joe Biden"],
        "retrieved_contexts": [["Paris", "France"], ["Joe Biden", "USA"]]
    }
    _dataset = EvalDataset(**data)
    
    # Single-turn evaluation
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness]
    eval_engine = ValidationEngine(dataset=_dataset, metrics=metrics)
    results, metrics, schema = eval_engine.evaluate()
    print("Single-turn evaluation results:")
    print(json.dumps(results, indent=2))
    print("JSON Schema:")
    print(json.dumps(schema, indent=2))