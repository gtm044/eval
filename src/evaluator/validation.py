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
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import json
import os

from src.evaluator.metrics import chunking, generation, retrieval
from src.evaluator.options import ValidationOptions
from src.data.dataset import EvalDataset

class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        options: ValidationOptions,
        output_dir: str = None
    ):
        self.dataset = dataset
        self.options = options
        self.output_dir = output_dir
        
    def evaluate(self):
        """
        Evaluate the performance of the RAG model.
        """
        # Calculate the metrics
        output = []
        metrics = self.calculate_metrics()
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
                if key=="avg_chunk_size":
                    data[key] = value
                elif key=="retrieval_accuracy":
                    data[key] = value
                else:
                    data[key] = value[i]
            output.append(data)
        
        return output
    
    def calculate_metrics(self):
        metrics = self.options.metrics
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
    eval = ValidationEngine(dataset=_dataset, options=_options)
    result = eval.evaluate()
    # Dump to a json file
    with open("output.json", "w") as f:
        json.dump(result, f, indent=4)