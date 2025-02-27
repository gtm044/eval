from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from src.evaluator.options import ValidationOptions

class ExperimentOptions(BaseModel):
    """
    Expeirment Configuration
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
        defualt = None,
        description="Language model used for generation"
    )
    

    