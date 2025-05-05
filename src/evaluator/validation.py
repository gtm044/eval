# src/evaluator/validation.py
from src.data.dataset import EvalDataset
from datasets import Dataset
from typing import List, Optional, Any, Dict, Callable
import json
import pandas as pd
import os
import ragas
from ragas import evaluate
from src.evaluator.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, avg_chunk_size, answer_similarity, context_similarity, context_score, llm_grading, tool_call_accuracy, tool_accuracy

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.metrics.base import Metric
from tqdm import tqdm
import inspect


class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        metrics: Optional[List[Any]] = None,
        segments: Optional[List[str]] = None,
        rubrics: Optional[List[str]] = None,
    ):
        """
        Args:
            dataset: The dataset containing questions, answers, responses, and contexts
            metrics: List of metrics to evaluate
            segments: List of segments to evaluate
        """
        self.dataset = dataset
        self.metrics = metrics
        self.segments = segments
        self.rubrics = rubrics
    
    
    def is_ragas_metric(self, metric):
        """
        Determine if a metric is a RAGAS metric or a custom metric.
        """
        if hasattr(metric, 'score_batch'): # Common ragas attributes
            return True
        if hasattr(metric, '__module__') and metric.__module__.startswith('ragas.'):
            return True
        if callable(metric):
            return False            
        return True
    
    
    def prepare_dataset(self):
        """
        Prepare a format suitable for both RAGAS and custom metrics.
        """
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
        if self.dataset.agent_responses is not None:
            dataset_dict["agent_responses"] = self.dataset.agent_responses
        if self.dataset.agent_tool_calls is not None:
            dataset_dict["agent_tool_calls"] = self.dataset.agent_tool_calls
        if self.dataset.agent_tool_outputs is not None:
            dataset_dict["agent_tool_outputs"] = self.dataset.agent_tool_outputs
        if self.dataset.reference_tool_calls is not None:
            dataset_dict["reference_tool_calls"] = self.dataset.reference_tool_calls
        if self.dataset.gt_answers is not None:
            dataset_dict["gt_answers"] = self.dataset.gt_answers
        if self.dataset.gt_tool_outputs is not None:
            dataset_dict["gt_tool_outputs"] = self.dataset.gt_tool_outputs
            
        golden_dataset = Dataset.from_dict(dataset_dict)
        return golden_dataset, dataset_dict
    
    
    def process_custom_metric(self, metric):
        """
        Run the evaluation on the custom metrics from the list of metrics.
        Args:
            metric: The custom metric to process
        Returns:
            tuple: (metric_name, result)
        """
        # Get the function params from the signature
        if callable(metric):
            try:
                sig = inspect.signature(metric)
                param_names = list(sig.parameters.keys())
                
                # Mapping the params to our attributes in the evaluation datset
                kwargs = {}
                param_mapping = {
                    "questions": self.dataset.questions,
                    "queries": self.dataset.questions,
                    "reference_contexts": self.dataset.reference_contexts,
                    "retrieved_contexts": self.dataset.retrieved_contexts,
                    "contexts": self.dataset.retrieved_contexts,
                    "responses": self.dataset.responses,
                    "model_answers": self.dataset.responses,
                    "answer": self.dataset.responses,
                    "ground_truths": self.dataset.answers,
                    "answers": self.dataset.answers,
                    "rubrics": self.rubrics,
                    "agent_responses": self.dataset.agent_responses,
                    "agent_tool_calls": self.dataset.agent_tool_calls,
                    "agent_tool_outputs": self.dataset.agent_tool_outputs,
                    "reference_tool_calls": self.dataset.reference_tool_calls,
                    "gt_answers": self.dataset.gt_answers,
                    "gt_tool_outputs": self.dataset.gt_tool_outputs
                }
                
                # For llm grading, we need the first answer from each list, 
                # Todo: integrate multiple answers for the llm grading
                if metric.__name__ == "llm_grading" and self.dataset.answers:
                    param_mapping["ground_truths"] = [answer_list[0] for answer_list in self.dataset.answers] if self.dataset.answers else []

                for param in param_names:
                    if param in param_mapping and param_mapping[param] is not None:
                        kwargs[param] = param_mapping[param]
                metric_name = metric.name if hasattr(metric, 'name') else metric.__name__
                print(f"Calculating {metric_name}...")
                
                # Call the metric function with the appropriate parameters
                result = metric(**kwargs)
                return metric_name, result
            except Exception as e:
                print(f"Error calculating {getattr(metric, 'name', metric.__name__)}: {e}")
                
        return None, None
    
    
    def process_ragas_metrics(self, metrics, golden_dataset):
        """
        Run the evaluation on the RAGAS metrics.       
        Args:
            metrics: List of RAGAS metrics to evaluate
            golden_dataset: The prepared dataset for evaluation
            
        Returns:
            pandas.DataFrame containing the results.
        """
        print(f"Evaluating with RAGAS metrics: {[getattr(m, 'name', str(m)) for m in metrics]}")
        try:
            results = ragas.evaluate(dataset=golden_dataset, metrics=metrics, show_progress=True)
            return results.to_pandas()
        except Exception as e:
            print(f"Error in RAGAS evaluation: {e}")
            return self.create_empty_dataframe()
    
    
    def create_empty_dataframe(self):
        """
        Create an empty df with the required attributes (if there are no RAGAS metrics).
        Returns:
            pandas.DataFrame: Empty dataframe with appropriate columns
        """
        df = pd.DataFrame(columns=["user_input", "retrieved_contexts", "response", "reference"])
        if self.dataset.questions:
            df["user_input"] = self.dataset.questions
        if self.dataset.retrieved_contexts:
            df["retrieved_contexts"] = self.dataset.retrieved_contexts
        if self.dataset.responses:
            df["response"] = self.dataset.responses
        if self.dataset.reference_contexts:
            df["reference"] = self.dataset.reference_contexts
        if self.dataset.agent_responses:
            df["agent_responses"] = self.dataset.agent_responses
        if self.dataset.agent_tool_calls:
            df["agent_tool_calls"] = self.dataset.agent_tool_calls
        if self.dataset.agent_tool_outputs:
            df["agent_tool_outputs"] = self.dataset.agent_tool_outputs
        if self.dataset.reference_tool_calls:
            df["reference_tool_calls"] = self.dataset.reference_tool_calls
        if self.dataset.gt_answers:
            df["gt_answers"] = self.dataset.gt_answers
        if self.dataset.gt_tool_outputs:
            df["gt_tool_outputs"] = self.dataset.gt_tool_outputs
            
        return df
    
    
    def determine_applicable_metrics(self, dataset_dict):
        """
        Determine applicable metrics based on available dataset fields if no metrics are provided.
        Args:
            dataset_dict: Dictionary containing dataset fields
        Returns:
            list: List of applicable metrics
        """
        applicable_metrics = []
        if "contexts" in dataset_dict and "question" in dataset_dict and "answer" in dataset_dict:
            applicable_metrics.extend([answer_relevancy])
        
        if "ground_truths" in dataset_dict and "answer" in dataset_dict and "question" in dataset_dict:
            applicable_metrics.extend([answer_correctness, answer_similarity])
        
        if "question" in dataset_dict and "reference" in dataset_dict and "contexts" in dataset_dict:
            applicable_metrics.extend([context_recall, context_precision])
        
        if not applicable_metrics:
            raise ValueError("No applicable metrics found for the provided dataset")
        return applicable_metrics
    
    
    def save_results(self, df):
        """
        Generate and save results to CSV and JSON files.
        Args:
            df: DataFrame containing evaluation results
        Returns:
            tuple: (results_dict, json_schema) - Dictionary of results and JSON schema
        """
        # Create results directory
        results_dir = ".results"
        os.makedirs(results_dir, exist_ok=True)
        # Save to CSV
        csv_filename = os.path.join(results_dir, "results.csv")
        df.to_csv(csv_filename, index=False)
        # Save to JSON
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
        for column in df.columns:
            json_schema["items"]["properties"][column] = {
                "type": "number" if df[column].dtype in ['float64', 'int64'] else "string",
            }
        
        print(f"Results saved to {csv_filename} and {json_filename}")
        return results_dict, json_schema
    
    
    def calculate_average_metrics(self, df, metrics):
        """
        Calculate the average of each metric in the results.
        Args:
            df: DataFrame containing evaluation results
            metrics: List of metrics used in evaluation
            
        Returns:
            dict: Dictionary of average metric values
        """
        avg_metrics = {}
        for metric in metrics:
            metric_name = getattr(metric, 'name', str(metric))
            if metric_name in df.columns:
                avg_metrics[metric_name] = df[metric_name].mean()
        
        return avg_metrics
            
            
    def evaluate(self):
        """
        Calculate metrics for the evaluation dataset.
        Returns:
            tuple: (results_dict, metrics_list, json_schema, avg_metrics)
        """
        # Prepare the dataset
        golden_dataset, dataset_dict = self.prepare_dataset()
        custom_metric_results = {}
        ragas_metrics = []
                    
        # Process metrics if provided
        if self.metrics:
            # Dynamically separate RAGAS metrics from custom metrics
            for metric in self.metrics:
                if not self.is_ragas_metric(metric):
                    print(f"Processing custom metric: {metric.name if hasattr(metric, 'name') else metric.__name__}")
                    metric_name, result = self.process_custom_metric(metric)
                    if metric_name and result is not None:
                        custom_metric_results[metric_name] = result
                else:
                    ragas_metrics.append(metric)
            
            # Process RAGAS metrics if there are any
            if ragas_metrics:
                df = self.process_ragas_metrics(ragas_metrics, golden_dataset)
            else:
                df = self.create_empty_dataframe()
        else:
            # Try to evaluate with default metrics
            try:
                results = ragas.evaluate(dataset=golden_dataset, show_progress=True)
                df = results.to_pandas()
            except Exception as e:
                print(f"Error with default metrics: {e}")
                # Determine applicable metrics based on available dataset fields
                applicable_metrics = self.determine_applicable_metrics(dataset_dict)
                
                print(f"Evaluating with applicable metrics: {[m.name for m in applicable_metrics]}")
                df = self.process_ragas_metrics(applicable_metrics, golden_dataset)
                # Update metrics list to applicable metrics for return value
                self.metrics = applicable_metrics
        
        # Add custom metric results to the DataFrame
        for metric_name, metric_result in custom_metric_results.items():
            df[metric_name] = metric_result
        # Add ground truth answers to the DataFrame if they exist
        if self.dataset.answers:
            df["ground_truth_answer"] = self.dataset.answers
        
        # Save results and get results dict and schema
        results_dict, json_schema = self.save_results(df)
        
        # Calculate average metrics
        metric_list = self.metrics if self.metrics else applicable_metrics
        avg_metrics = self.calculate_average_metrics(df, metric_list)
        
        return results_dict, metric_list, json_schema, avg_metrics
        
        
if __name__=='__main__':
    # Example
    # data = {
    #     "questions": ["What is the capital of France?", "Who is the president of the USA?"],
    #     "answers": [["Paris", "France"], ["Joe Biden", "USA"]],
    #     "responses": ["Capital of france is Paris", "President of the USA is Joe Biden"],
    #     "reference_contexts": ["Paris is the capital of France", "Joe Biden is the 46th president of the USA"],
    #     "retrieved_contexts": [["Paris is the capital of France", "France is in Europe"], ["Joe Biden is the 46th president of the USA", "The USA is a country in North America"]]
    # }
    # _dataset = EvalDataset(**data)
    
    # # Define custom metric functions
    # def response_length(responses):
    #     """
    #     Custom metric that calculates the length of each response
    #     """
    #     print("Calculating response length...")
    #     response_lengths = [len(response) for response in responses]
    #     return response_lengths
    
    # def context_to_response_ratio(responses, retrieved_contexts):
    #     """
    #     Custom metric that calculates the ratio of context length to response length
    #     """
    #     print("Calculating context to response ratio...")
    #     ratios = []
    #     for i, response in enumerate(responses):
    #         if i < len(retrieved_contexts):
    #             # Calculate total length of all contexts for this response
    #             total_context_length = sum(len(ctx) for ctx in retrieved_contexts[i])
    #             response_length = len(response)
    #             ratio = total_context_length / response_length if response_length > 0 else 0
    #             ratios.append(ratio)
    #         else:
    #             ratios.append(0)
    #     return ratios

    
    # metrics = [
    #     # Custom metrics
    #     response_length,
    #     context_to_response_ratio,
    #     # System-implemented metrics that work reliably (not RAGAS metrics)
    #     context_similarity,
    #     context_score,
    #     # Ragas metrics
    #     answer_correctness,
    #     answer_similarity,
    #     answer_relevancy,
    # ]
    
    # # Create the validation engine with the metrics
    # eval_engine = ValidationEngine(dataset=_dataset, metrics=metrics)
    
    # # Run the evaluation
    # results, used_metrics, schema, avg_metrics = eval_engine.evaluate()
    
    # print("\nEvaluation results with custom and system metrics:")
    # print(f"Used metrics: {[getattr(m, 'name', str(m)) for m in used_metrics]}")
    # print("\nAverage metrics:")
    # print(json.dumps(avg_metrics, indent=2))
    
    # print("\nSample results (first data point):")
    # print(json.dumps(results[0], indent=2))
    
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
    metrics = [tool_call_accuracy, tool_accuracy]
    eval_engine = ValidationEngine(dataset=dataset, metrics=metrics)
    results, used_metrics, schema, avg_metrics = eval_engine.evaluate()
    print(results)