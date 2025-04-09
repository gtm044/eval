# src/evaluator/validation.py
from src.data.dataset import EvalDataset
from datasets import Dataset
from typing import List, Optional, Any
import json
import pandas as pd
import os
import ragas
from ragas import evaluate
from src.evaluator.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, avg_chunk_size, answer_similarity, context_similarity, context_score, llm_grading
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
        context_similarity_result = None
        context_score_result = None
        llm_grading_result = None
        
        self.temp_metrics = self.metrics.copy()
                    
        # Check if the custom metrics are present, if so calculate the metrics and remove them from the list
        if self.metrics:
            # Create a list to track metrics to remove
            metrics_to_remove = []
            
            # First identify all metrics to remove
            for i, metric in enumerate(self.metrics):
                if metric.name == "avg_chunk_size":
                    metrics_to_remove.append(i)
                    print("Calculating average chunk size...")
                    avg_chunk_size_result = avg_chunk_size([context for context in tqdm(self.dataset.reference_contexts, desc="Processing chunks")])
                elif metric.name == "context_similarity":
                    metrics_to_remove.append(i)
                    print("Calculating context similarity...")
                    context_similarity_result = context_similarity(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
                elif metric.name == "context_score":
                    metrics_to_remove.append(i)
                    print("Calculating context score...")
                    context_score_result = context_score(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
                elif metric.name == "llm_grading":
                    metrics_to_remove.append(i)
                    print("Calculating llm grading...")
                    # Extract the first string from each list in the answers list of lists
                    prime_answers = [answer_list[0] for answer_list in self.dataset.answers] if self.dataset.answers else []
                    llm_grading_result = llm_grading(queries=self.dataset.questions, ground_truths=prime_answers, model_answers=self.dataset.responses)
                    
            # Remove metrics in reverse order to avoid index shifting
            for i in sorted(metrics_to_remove, reverse=True):
                self.temp_metrics.pop(i)
            
            if len(self.temp_metrics)>0:
                results = ragas.evaluate(dataset=golden_dataset, metrics=self.temp_metrics, show_progress=True)
            # Fix this headache, if ragas metrics are not provided then we dont have a results object, can we create a rags result object with nothing?
            else:
                # Create a pandas dataframe with the same structure as the results dataframe
                results = pd.DataFrame(columns=["user_input", "retrieved_contexts", "response", "reference"])
                results["user_input"] = self.dataset.questions
                results["retrieved_contexts"] = self.dataset.retrieved_contexts
                results["response"] = self.dataset.responses
                results["reference"] = self.dataset.reference_contexts
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
                    applicable_metrics.extend([answer_relevancy, faithfulness])
                
                # Check for ground truth-based metrics
                if "ground_truths" in dataset_dict and "answer" in dataset_dict and "question" in dataset_dict:
                    applicable_metrics.extend([answer_correctness, answer_similarity])
                
                # Check for reference-based metrics
                if "question" in dataset_dict and "reference" in dataset_dict and "contexts" in dataset_dict:
                    applicable_metrics.extend([context_recall, context_precision])
                
                if not applicable_metrics:
                    raise ValueError("No applicable metrics found for the provided dataset")
                
                print(f"Evaluating with applicable metrics: {[m.name for m in applicable_metrics]}")
                results = ragas.evaluate(dataset=golden_dataset, metrics=applicable_metrics, show_progress=True)
        

        # Convert to pandas DataFrame if results is not of type dataframe
        df = results.to_pandas() if not isinstance(results, pd.DataFrame) else results
        
        if self.dataset.answers is not None:
            df["ground_truth_answer"] = self.dataset.answers
        
        # Add the avg chunk size result to the results
        if avg_chunk_size_result is not None:
            df["avg_chunk_size"] = avg_chunk_size_result
        
        # Add the context similarity result to the results
        if context_similarity_result is not None:
            df["context_similarity"] = context_similarity_result
        
        # Add the context score result to the results
        if context_score_result is not None:
            df["context_score"] = context_score_result
            
        # Add the llm grading result to the results
        if llm_grading_result is not None:
            df["llm_grading"] = llm_grading_result
            
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
            
        # For the metrics that are calculated, find the average of each metric
        avg_metrics = {}
        if self.metrics:
            for metric in self.metrics:
                if metric.name in df.columns:
                    avg_metrics[metric.name] = df[metric.name].mean()
        else:
            for metric in applicable_metrics:
                if metric.name in df.columns:
                    avg_metrics[metric.name] = df[metric.name].mean()
        
        print(f"Results saved to {csv_filename} and {json_filename}")
        if self.metrics:
            return results_dict, self.metrics, json_schema, avg_metrics
        else:
            return results_dict, applicable_metrics, json_schema, avg_metrics
        
        
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
    # metrics = [context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size, context_similarity, context_score]
    metrics = [llm_grading]
    eval_engine = ValidationEngine(dataset=_dataset, metrics=metrics) # Dont provide metrics if you want to use the default metrics
    results, metrics, schema, avg_metrics = eval_engine.evaluate()
    print("Single-turn evaluation results:")
    print(json.dumps(results, indent=2))
    print("JSON Schema:")
    print(json.dumps(schema, indent=2))
    print("Average metrics:")
    print(json.dumps(avg_metrics, indent=2))
    
    ## Note: The avg_chunk_size will be the same for all data points as it is a normalized index. 
    ## Ranges from -inf to 1 (Higher is better)
    ## Any score above 0.5 is acceptable.
    
    # How to let users add their own metrics?
    # 1. Let users define a function according to a predefined schema
    # 2. Add the function to the metrics list
    # 3. Provide a decorator to the function so that it can be used as a metrics
    # 4. Add the function to @src.evaluator.metrics.definitions