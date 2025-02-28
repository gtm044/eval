# src/evaluator/options.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class ValidationOptions(BaseModel):
    """
    Validation Options
    
    This class defines the configuration options for validating RAG systems, including:
    - Metrics to calculate for evaluation
    - Specific RAG segments to evaluate (chunking, retrieval, generation)
    - Report generation settings
    """
    metrics: Optional[List[str]] = Field(
        default = None,
        description="List of metrics to calculate",
        validate_default=True
    )
    segments: Optional[List[str]] = Field(
        default = None,
        description="List of particular segments of the RAG to evaluate",
        validate_default=True
    )
    generateReport: bool = Field(
        default = False,
        description="Generate a report of the evaluation"
    )
    
    # Validate if the metrics provided is present in the list of implemented metrics
    @field_validator("metrics", mode="before")
    def check_metrics(cls, v):
        # List of all metrics currently implemented in the system
        implemented_metrics = [
            "avg_chunk_size",      # Measures average size of document chunks
            "jaccard_index",       # Measures overlap between sets
            "context_score",       # Evaluates relevance of retrieved context
            "embedding_similarity", # Measures vector similarity between embeddings
            "named_entity_score",  # Evaluates named entity matching
            "retrieval_accuracy",  # Measures accuracy of retrieval system
            "bleu_score",          # Evaluates text generation quality
            "rouge_score",         # Measures summary quality
            "faithfulness",        # Evaluates factual consistency
            "response_similarity"  # Measures similarity between responses
        ]
        
        # Ensure input is a list
        if not isinstance(v, list):
            raise ValueError("Metrics must be a list.")
        
        # Validate each metric against implemented options
        for metric in v:
            if metric not in implemented_metrics:
                raise ValueError(f"Metric {metric} is not implemented yet. Choose any metric from {implemented_metrics}")
        return v
    
if __name__ == '__main__':
    # Example usage of ValidationOptions
    options = ValidationOptions(
        metrics=[
            "avg_chunk_size", 
            "jaccard_index", 
            "context_relevance",  # Note: This will raise an error as it's not implemented
            "context_recall",     # Note: This will raise an error as it's not implemented
            "context_precision",  # Note: This will raise an error as it's not implemented
            "retrieval_similarity", # Note: This will raise an error as it's not implemented
            "named_entity_score"
        ], 
        segments=["chunking", "retrieval", "generation"], 
        generateReport=True
    )
    
    print(options)