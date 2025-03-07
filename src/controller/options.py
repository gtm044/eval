from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from ragas.metrics.base import Metric
import uuid

class ExperimentOptions(BaseModel):
    """
    Experiment Configuration
    
    This class defines the configuration options for an experiment, including:
    - Experiment identifier
    - Text chunking parameters (size and overlap)
    - Embedding model configuration
    - Language model selection
    """
    experiment_id: Optional[str] = Field(
        default = str(uuid.uuid4()),
        description="Experiment ID"
    )
    dataset_id: Optional[str] = Field(
        default = None,
        description="ID of the evaluation dataset (EvalDataset.dataset_id)"
    )
    metrics: Optional[List] = Field(
        default = None,
        description="""
        Metrics to evaluate
            Implemented metrics:
            - context_precision
            - context_recall
            - answer_relevancy
            - faithfulness
            - answer_correctness
            - avg_chunk_size
        """
    )
    segments: Optional[List[str]] = Field( # Yet to implement
        default = None,
        description="Segments to evaluate"
    )
    chunk_size: Optional[int] = Field(
        default = None,
        description="Chunk size"
    )
    chunk_overlap: Optional[int] = Field(
        default = None,
        description="Chunk overlap"
    )
    embedding_model: Optional[str] = Field(
        default = None,
        description="Embedding model used for semantic search"
    )
    embedding_dimension: Optional[int] = Field(
        default = None,
        description="Embedding dimension"
    )
    llm_model : Optional[str] = Field(
        default = None,  
        description="Language model used for generation"
    )
    
    # Check whether the metrics provided belong to the list of implemented metrics
    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v, info):
        metric_names = [metric.name for metric in v]
        for metric_name in metric_names:
            if metric_name not in ["context_precision", "context_recall", "answer_relevancy", "faithfulness", "answer_correctness", "avg_chunk_size"]:  
                raise ValueError(f"{metric_name} doesn't exist.")
            
    # Check whether the dataset contains the required params for calculating the metrics
    # Allows users to debug the dataset before running the experiment
    @field_validator("dataset_id")
    @classmethod
    def validate_dataset(cls, v, info):
        if v is None:
            raise ValueError("Dataset ID is required")
        return v
        

if __name__=='__main__':
    from src.evaluator.metrics import faithfulness, avg_chunk_size
    try:
        experimentOptions = ExperimentOptions(
            metrics = [faithfulness, avg_chunk_size]
        )
    except ValueError as e:
        print(e)
        
    metrics = [faithfulness, avg_chunk_size]
    for metric in metrics:
        print(metric.name)