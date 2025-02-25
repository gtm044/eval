# src/evaluator/options.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class ValidationOptions(BaseModel):
    """
    Validation Options
    """
    experiment_id: str = Field(
        ...,
        description="Experiment ID"
    )
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
        implemented_metrics = [
            "avg_chunk_size", "jaccard_index", "context_score", "embedding_similarity", "named_entity_score"
        ]
        if not isinstance(v, list):
            raise ValueError("Metrics must be a list.")
        
        for metric in v:
            if metric not in implemented_metrics:
                raise ValueError(f"Metric {metric} is not implemented yet. Choose any metric from {implemented_metrics}")
        return v
    
if __name__ == '__main__':
    options = ValidationOptions(experiment_id="1234", metrics=["avg_chunk_size", "jaccard_index", "context_relevance", "context_recall", "context_precision", "retrieval_similarity", "named_entity_score"], segments=["chunking", "retrieval", "generation"], generateReport=True)
    
    print(options)