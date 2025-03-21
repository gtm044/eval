# Create a experiment management system that manages evalautions as experiments
import os
import json
from typing import List, Optional
import pandas as pd
from datetime import datetime
from src.controller.options import ExperimentOptions
from src.data.load import LoadOperator
from src.data.dataset import EvalDataset
from couchbase.kv_range_scan import PrefixScan
from src.evaluator.validation import ValidationEngine

class Experiment:
    def __init__(self, dataset: Optional[EvalDataset] = None, options: Optional[ExperimentOptions] = None):
        """
        Initialize an Experiment instance with configuration options and evaluation results.
        
        Args:
            options: Configuration parameters for the experiment
        """
        # Couchbase cluster credentials
        self.bucket = os.getenv("bucket")
        self.scope = os.getenv("scope")
        self.collection = os.getenv("collection")
        self.cluster_url = os.getenv("cluster_url")
        self.username = os.getenv("cb_username")
        self.password = os.getenv("cb_password")
        
        # If experiment options are not provided, the class acts as a retriever
        if options is None:
            return
        
        self.options = options
        
        # Load the dataset
        if dataset is not None:
            self.dataset = dataset
        else:
            if self.options.dataset_id is None:
                raise ValueError("Dataset ID must be provided in experiment options")
            self.load_operator = LoadOperator()
            self.dataset = self.load_operator.retrieve_docs(self.options.dataset_id)
        
            if not isinstance(self.dataset, EvalDataset):
                raise ValueError(f"Failed to load dataset with ID: {self.options.dataset_id}")
        
        # Create validation engine and run evaluation
        validation_engine = ValidationEngine(
            dataset=self.dataset,
            metrics=self.options.metrics,
            segments=self.options.segments
        )
        
        # Run the evaluation
        self.output, self.metrics, _, self.avg_metrics = validation_engine.evaluate()
        
        # Rename the .results directory created by the the validationEngine to ".results-experiment_id"
        # If the directory exits, don't rename it
        results_dir = ".results"
        new_results_dir = f".results-{self.options.experiment_id}"
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            if not os.path.exists(new_results_dir):
                os.rename(results_dir, new_results_dir)
        
        # Metrics retrieved from the validation engine are of type ragas metric object, get their names
        self.metric_names = [metric.name for metric in self.metrics]
        
        # Create the experiment metadata with configuration and results summary
        # Create base metadata with standard fields
        self.metadata = {
            "experiment_id": self.options.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_size": self.options.chunk_size,
            "chunk_overlap": self.options.chunk_overlap,
            "embedding_model": self.options.embedding_model,
            "embedding_dimension": self.options.embedding_dimension,
            "llm_model": self.options.llm_model,
            "metrics": self.metric_names,
            "dataset_size": len(self.output),
            "dataset_id": self.options.dataset_id,
            "avg_metrics": self.avg_metrics
        }
        
        # Add any custom fields from options to metadata
        # BaseModel objects don't use __dict__ directly, use model_dump() instead
        options_dict = self.options.model_dump()
        for field_name, field_value in options_dict.items():
            if (field_name not in self.metadata and 
                field_name != "metrics" and 
                field_name != "experiment_id" and
                field_name != "dataset_id" and
                field_name != "chunk_size" and
                field_name != "chunk_overlap" and
                field_name != "embedding_model" and
                field_name != "embedding_dimension" and
                field_name != "llm_model" and
                field_name != "segments"):
                self.metadata[field_name] = field_value
        
        # Save the metadata to the results directory
        results_dir = f".results-{self.options.experiment_id}"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
            
            
    def load_to_couchbase(self,  collection=None):
        """
        Load the experiment data to Couchbase database.
        
        Args:
            collection: Optional custom collection name to use instead of default
        """
        # If a separate collection is defined by the user, use that collection
        if collection is not None:
            self.collection = collection
            
        object_id = 1
        load_operator = LoadOperator()
        cb = load_operator.connect(collection=self.collection)
        cb_coll = cb.scope(self.scope).collection(self.collection)
        batch_exceptions = []
        batch_size = 100  
        
        # Create Couchbase objects for the individual data points and load them in batches
        for i in range(0, len(self.output), batch_size):
            data = dict()
            for object in self.output[i:i+batch_size]:
                object["metadata"] = self.metadata  
                data[f"{self.options.experiment_id}_{object_id}"] = object
                object_id += 1
            results = cb_coll.upsert_multi(data)
            # Track any exceptions during batch upload
            if len(results.exceptions) > 0:
                batch_exceptions.append(results.exceptions)
        
        # Raise exception if any batch upload failed
        if len(batch_exceptions) > 0:
            raise Exception(batch_exceptions)
        else:
            print(f"Experiment data loaded successfully with experiment id: {self.options.experiment_id}")
        
    
    def retrieve(self, experiment_id, collection=None):
        """
        Retrieve an experiment from Couchbase database.
        
        Args:
            experiment_id: Identifier for the experiment to retrieve
            collection: Optional custom collection name to use instead of default
        """
        results_dir = f".results-{experiment_id}"
        result_json_path = os.path.join(results_dir, "results.json")
        
        if collection is not None:
            self.collection = collection
        content = []
        
        # Get the connection to couchbase cluster
        load_operator = LoadOperator()
        cb = load_operator.connect(collection=self.collection)
        cb_coll = cb.scope(self.scope).collection(self.collection)
        
        # Create a prefix scan using the experiment id
        prefix = f"{experiment_id}_"
        experiment_docs = cb_coll.scan(PrefixScan(prefix))
        for doc in experiment_docs:
            content.append(doc.content_as[dict])
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the results as JSON, CSV and metadata
        with open(result_json_path, "w") as f:
            # Remove metadata from each item before saving
            content_without_metadata = []
            for item in content:
                item_copy = item.copy()
                if "metadata" in item_copy:
                    del item_copy["metadata"]
                content_without_metadata.append(item_copy)
            json.dump(content_without_metadata, f, indent=4)
        
        # Save as CSV
        df = pd.DataFrame(content)
        if "metadata" in df.columns:
            df = df.drop(columns=["metadata"])
        df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
        
        # Save metadata
        if content and "metadata" in content[0]:
            with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
                json.dump(content[0]["metadata"], f, indent=4)
        
        print(f"Retrieved experiment data saved to {results_dir}")    
    
if __name__=='__main__':
    from src.evaluator.metrics import faithfulness, context_precision
    # Example usage for the Experiment class  
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": [["Paris", "France"], ["Washington", "USA"]],
        "responses": ["Paris", "Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "The USA is a country in North America"],
        "retrieved_contexts": [["Paris is the capital of France", "France is a country in Europe"], ["Washington is the capital of the USA", "The USA is a country in North America"]]
    }
    dataset = EvalDataset(**data)
    dataset.to_json("test.json")
    loader = LoadOperator(dataset=dataset, dataset_description="Test dataset")
    loader.load_docs()
      
    # Create experiment options
    options = ExperimentOptions(
        experiment_id="123",
        dataset_id=dataset.dataset_id,
        metrics=[context_precision, faithfulness],
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        llm_model="gpt-4o-mini",
        embedding_dimension=3072
    )
    
    # Initialize experiment with options
    experiment = Experiment(options=options)
    experiment.load_to_couchbase(collection="experiment2")
    
    # # Test retrieve functionality
    experiment = Experiment()
    print("Testing retrieve functionality...")
    experiment.retrieve(experiment_id="123", collection="experiment2")
    
    print("Retrieved experiment data saved to .results-123 directory")
     