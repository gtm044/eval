# src/evaluator/validation.py
from src.data.dataset import EvalDataset
from datasets import Dataset
from typing import List, Optional, Any
import json
import pandas as pd
import os
import ragas
from ragas import evaluate
from src.evaluator.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, avg_chunk_size
# from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics.base import Metric
from tqdm import tqdm


class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        metrics: Optional[List[Any]] = None,
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
        dataset_dict = {}
        
        if self.dataset.questions is not None:
            dataset_dict["question"] = self.dataset.questions
        
        if self.dataset.responses is not None:
            dataset_dict["answer"] = self.dataset.responses
        
        if self.dataset.retrieved_contexts is not None:
            dataset_dict["contexts"] = self.dataset.retrieved_contexts
        
        if self.dataset.answers is not None:
            dataset_dict["ground_truths"] = self.dataset.answers
        
        if self.dataset.reference_contexts is not None:
            dataset_dict["reference"] = self.dataset.reference_contexts
        
        golden_dataset = Dataset.from_dict(dataset_dict)
        
        avg_chunk_size_result = None
            
        # Check if avg_chunk_size is in the metrics list, if present, remove it calculate the avg chunk size
        if self.metrics:
            for i, metric in enumerate(self.metrics):
                if metric.name == "avg_chunk_size":
                    self.metrics.pop(i)
                    print("Calculating average chunk size...")
                    avg_chunk_size_result = avg_chunk_size([context for context in tqdm(self.dataset.reference_contexts, desc="Processing chunks")])
            results = ragas.evaluate(dataset=golden_dataset, metrics=self.metrics, show_progress=True)
            
        else:
            # Try to evaluate with default metrics, but handle errors for incompatible metrics
            try:
                results = ragas.evaluate(dataset=golden_dataset, show_progress=True)
            except Exception as e:
                print(f"Error with default metrics: {e}")
                # Determine applicable metrics based on available dataset fields
                applicable_metrics = []
                
                # Check for context-based metrics
                if "contexts" in dataset_dict and "question" in dataset_dict and "answer" in dataset_dict:
                    from ragas.metrics import answer_relevancy, faithfulness
                    applicable_metrics.extend([answer_relevancy, faithfulness])
                
                # Check for ground truth-based metrics
                if "ground_truths" in dataset_dict and "answer" in dataset_dict and "reference" in dataset_dict and "question" in dataset_dict:
                    from ragas.metrics import answer_correctness
                    applicable_metrics.append(answer_correctness)
                
                # Check for reference-based metrics
                if "question" in dataset_dict and "reference" in dataset_dict and "contexts" in dataset_dict:
                    from ragas.metrics import context_recall
                    applicable_metrics.extend([context_recall, context_precision])
                
                if not applicable_metrics:
                    raise ValueError("No applicable metrics found for the provided dataset")
                
                print(f"Evaluating with applicable metrics: {[m.name for m in applicable_metrics]}")
                results = ragas.evaluate(dataset=golden_dataset, metrics=applicable_metrics, show_progress=True)
        
        # Convert to pandas DataFrame
        df = results.to_pandas()
        
        # Add the avg chunk size result to the results
        if avg_chunk_size_result is not None:
            df["avg_chunk_size"] = avg_chunk_size_result
        
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
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size]
    eval_engine = ValidationEngine(dataset=_dataset) # Dont provide metrics if you want to use the default metrics
    results, metrics, schema = eval_engine.evaluate()
    print("Single-turn evaluation results:")
    print(json.dumps(results, indent=2))
    print("JSON Schema:")
    print(json.dumps(schema, indent=2))
    
    ## Note: The avg_chunk_size will be the same for all data points as it is a normalized index. 
    ## Ranges from -inf to 1 (Higher is better)
    ## Any score above 0.5 is acceptable.
    
    # How to let users add their own metrics?
    # 1. Let users define a function according to a predefined schema
    # 2. Add the function to the metrics list
    # 3. Provide a decorator to the function so that it can be used as a metrics
    # 4. Add the function to @src.evaluator.metrics.definitions