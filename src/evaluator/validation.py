# src/evaluator/validation.py

## Creates a validation engine to evaluate the performance of the RAG model.
# Inputs:
# - EvalDataset method
# - ValidationOptions method
# - output directory to save the results
# Outputs:
# - list of dictionaries containing the data points and the corresponsing evaluation metrics.
# - saved as a json file in a given output directory or to a csv file
# - A report generated if the generateReport option is set to True
# - Finally an index created using weighted average of the metrics calculated -> provides an overall evaluation of the RAG model.
from email.headerregistry import HeaderRegistry
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json
import os
import pandas as pd

from src.evaluator.metrics import chunking, generation, retrieval
from src.evaluator.options import ValidationOptions
from src.data.dataset import EvalDataset

class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        options: ValidationOptions,
    ):
        self.dataset = dataset
        self.options = options
        
    def evaluate(self):
        """
        Evaluate the performance of the RAG model.
        """
        # Calculate the metrics
        output = []
        list_of_metrics = []
        
        # If segments are provided, calculate the metrics for the segments
        if self.options.segments:
            for segment in self.options.segments:
                if segment == "chunking":
                    list_of_metrics.append("avg_chunk_size")
                elif segment == "retrieval":
                    list_of_metrics.extend(["context_score", "embedding_similarity", "named_entity_score", "retrieval_accuracy"])
                elif segment == "generation":
                    list_of_metrics.extend(["bleu_score", "rouge_score", "faithfulness", "response_similarity"])
        else:
            list_of_metrics = self.options.metrics
            
        metrics = self.calculate_metrics(list_of_metrics)
    
        # Create a list of dictionaries containing the data points and the corresponsing evaluation metrics
        # Initially add a dictionary for the average metrics (avg_chunk_size, retrieval_accuracy)
        output.append({
            "avg_chunk_size": metrics["avg_chunk_size"],
            "retrieval_accuracy": metrics["retrieval_accuracy"],
            "avg_context_score": [sum(x) / len(x) for x in zip(*metrics["context_score"])],
            "avg_embedding_similarity": sum(metrics["embedding_similarity"]) / len(metrics["embedding_similarity"]),
            "avg_named_entity_score": sum(metrics["named_entity_score"]) / len(metrics["named_entity_score"]),
            "avg_bleu_score": sum(metrics["bleu_score"]) / len(metrics["bleu_score"]),
            "avg_rouge_score": [sum(x) / len(x) for x in zip(*metrics["rouge_score"])],
            "avg_faithfulness": sum(metrics["faithfulness"]) / len(metrics["faithfulness"]),
            "avg_response_similarity": sum(metrics["response_similarity"]) / len(metrics["response_similarity"])
        })
        
        for i in range(len(self.dataset.questions)):
            data = {
                "question": self.dataset.questions[i],
                "answer": self.dataset.answers[i],
                "response": self.dataset.responses[i],
                "reference_context": self.dataset.reference_contexts[i],
                "retrieved_context": self.dataset.retrieved_contexts[i]
            }
            for key, value in metrics.items():
                # Handle metrics with just one single value
                if key=="avg_chunk_size" or key=="retrieval_accuracy":
                    continue   
                else:
                    data[key] = value[i]
            output.append(data)
            
            
        # # Dump and save the output to a json file
        # if self.output_dir:
        #     with open(os.path.join(self.output_dir, "output.json"), "w") as f:
        #         json.dump(output[1:], f, indent=4)
        #     with open(os.path.join(self.output_dir, "averaged_output.json"), "w") as f:
        #         json.dump(output[:1], f, indent=4)
        #     with open(os.path.join(self.output_dir, "output.csv"), "w") as f:
        #         df = pd.json_normalize(output[1:])
        #         df.to_csv(f, index=False)
        #     with open(os.path.join(self.output_dir, "averaged_output.csv"), "w") as f:
        #         df = pd.json_normalize(output[:1])
        #         df.to_csv(f, index=False)
        
        return output, list_of_metrics
    
    def calculate_metrics(self, metrics):
        scores = {}
        for metric in metrics:
            if metric == "avg_chunk_size":
                scores["avg_chunk_size"] = chunking.avg_chunk_size(self.dataset.reference_contexts)
            elif metric == "jaccard_index":
                pass # Not going to implement this as of now, need to figure out how the chunk ground truth is generated.
            elif metric == "context_score":
                scores["context_score"] = retrieval.context_score(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
            elif metric == "embedding_similarity":
                scores["embedding_similarity"] = retrieval.embedding_similarity(self.dataset.questions, self.dataset.retrieved_contexts)
            elif metric == "named_entity_score":
                scores["named_entity_score"] = retrieval.named_entity_score(self.dataset.questions, self.dataset.retrieved_contexts)
            elif metric == "retrieval_accuracy":
                scores["retrieval_accuracy"] = retrieval.retrieval_accuracy(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
            elif metric == "bleu_score":
                scores["bleu_score"] = generation.bleu_score(self.dataset.answers, self.dataset.responses)
            elif metric == "rouge_score":
                scores["rouge_score"] = generation.rouge_score(self.dataset.answers, self.dataset.responses)
            elif metric == "faithfulness":
                scores["faithfulness"] = generation.faithfulness(self.dataset.retrieved_contexts, self.dataset.responses)
            elif metric == "response_similarity":
                scores["response_similarity"] = generation.response_similarity(self.dataset.answers, self.dataset.responses)
            else:
                raise ValueError(f"Metric {metric} is not implemented yet.")
        return scores
    
    # Not completed yet
    def calculate_index(self, avg_scores):
        """
        Calculate the index based on the weighted average of the metrics.
        """
        weight = 0.5
        chunking_index = (weight * avg_scores["avg_chunk_size"] + (1 - weight) * avg_scores["retrieval_accuracy"]) / 2
        retrieval_index = (weight * avg_scores["context_score"] + weight * avg_scores["embedding_similarity"] + weight * avg_scores["named_entity_score"] + weight * avg_scores["retrieval_accuracy"]) / 4
        generation_index = (weight * avg_scores["bleu_score"] + weight * avg_scores["rouge_score"] + weight * avg_scores["faithfulness"] + weight * avg_scores["response_similarity"]) / 4
        return {
            "chunking_index": chunking_index,
            "retrieval_index": retrieval_index,
            "generation_index": generation_index
        }
        
        
        
if __name__=='__main__':
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": ["Paris is the capital of france", "Joe Biden is the president of the USA"],
        "responses": ["Capital of france is Paris", "President of the USA is Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"],
        "retrieved_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"]
    }
    _dataset = EvalDataset(**data)
    _options = ValidationOptions(
        experiment_id="1234",
        metrics = [
            "avg_chunk_size", "context_score", "embedding_similarity", "named_entity_score", "retrieval_accuracy", "bleu_score", "rouge_score", "faithfulness", "response_similarity"
        ],
        generateReport=False
    )
    eval = ValidationEngine(dataset=_dataset, options=_options, output_dir="output_data/")
    result = eval.evaluate()