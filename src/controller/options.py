from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from ragas.metrics.base import Metric
import uuid
from src.evaluator.metrics import llm_grading
# Add langgraph_logs to the options.
# If the logs are set to true, then instead of eval dataset, the logs are taken into consideration
# The logs can be pushed to the couchbase cluster.
# A vaidation engine specificaly for agentic applicaitons that takes in the langgrah logs and validates them.
# Again we would want to have dataset object to structure the logs and for the references
# Question: How will the user provide the references, what structure will the evaluation dataset be in?


class ExperimentOptions(BaseModel):
    """
    Experiment Configuration
    
    This class defines the configuration options for an experiment, including:
    - Experiment identifier
    - Text chunking parameters (size and overlap)
    - Embedding model configuration
    - Language model selection
    
    Users can also add custom fields as needed for their specific experiment requirements.
    """
    experiment_id: Optional[str] = Field(
        default = str(uuid.uuid4()),
        description="Experiment ID"
    )
    dataset_id: Optional[str] = Field(
        default = None,
        description="ID of the evaluation dataset (EvalDataset.dataset_id)"
    )
    langgraph: Optional[bool] = Field(
        default = False,
        description="If true, evaluation is performed on the langgraph logs for agentic applications"
    ) 
    metrics: Optional[List[Any]] = Field(
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
            - context_similarity
            - context_score
            - llm_grading
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
    
    # Allow arbitrary fields to be added
    class Config:
        extra = "allow"
    
    # Check whether the metrics provided belong to the list of implemented metrics
    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v, info):
        if v is None:
            return v
        
        # Get metric names - handle both direct metric objects and redefined metrics
        metric_names = []
        for metric in v:
            if hasattr(metric, 'name'):
                metric_names.append(metric.name)
        
        # Validate metric names
        for metric_name in metric_names:
            if metric_name not in ["context_precision", "context_recall", "answer_relevancy", "faithfulness", "answer_correctness", "avg_chunk_size", "context_similarity", "context_score", "named_entity_score", "semantic_similarity", "llm_grading"]:  
                raise ValueError(f"{metric_name} doesn't exist.")
        
        return v
            

if __name__=='__main__':
    from src.evaluator.metrics import faithfulness, avg_chunk_size
    try:
        # Create experiment options with custom fields
        experimentOptions = ExperimentOptions(
            experiment_id="test_exp_001",
            dataset_id="sample_dataset_123",
            metrics=[llm_grading],
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_dimension=768,
            llm_model="gpt-4o-mini",
            custom_field1="custom value",  # Testing arbitrary field
            custom_field2=42              # Testing arbitrary field
        )
        print(f"Experiment ID: {experimentOptions.experiment_id}")
        print(f"Dataset ID: {experimentOptions.dataset_id}")
        print(f"Custom field 1: {experimentOptions.custom_field1}")
        print(f"Custom field 2: {experimentOptions.custom_field2}")
        print(f"Metrics: {experimentOptions.metrics}")
    except ValueError as e:
        print(f"Error: {e}")