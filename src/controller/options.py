from pydantic import BaseModel, Field
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
    metrics: Optional[List[Metric] | List[str]] = Field(
        default = None,
        description="""
        Metrics to evaluate
            Implemented metrics:
            - context_precision
            - context_recall
            - answer_relevancy
            - faithfulness
            - answer_correctness
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