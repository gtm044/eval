from pydantic import BaseModel, Field
from typing import Optional


class ExperimentOptions(BaseModel):
    """
    Experiment Configuration
    
    This class defines the configuration options for an experiment, including:
    - Experiment identifier
    - Text chunking parameters (size and overlap)
    - Embedding model configuration
    - Language model selection
    """
    experiment_id: str = Field(
        ...,
        description="Experiment ID"
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