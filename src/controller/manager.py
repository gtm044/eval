# Create a experiment management system that manages evalautions as experiments
import os
import json
from typing import List
import pandas as pd
from datetime import datetime, timedelta
from src.controller.options import ExperimentOptions

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, TLSVerifyMode
from couchbase.exceptions import CouchbaseException, DocumentNotFoundException


class Experiment:
    def __init__(self, options: ExperimentOptions, evaluation_output: List[dict]):
        """
        Initialize an Experiment instance with configuration options and evaluation results.
        
        Args:
            options: Configuration parameters for the experiment
            evaluation_output: List containing evaluation results and metrics
        """
        self.options = options
        self.output, self.metrics = evaluation_output
        # Couchbase connection credentials from environment variables
        self.bucket = os.getenv("bucket")
        self.scope = os.getenv("scope")
        self.collection = os.getenv("collection")
        self.cluster_url = os.getenv("cluster_url")
        self.username = os.getenv("cb_username")
        self.password = os.getenv("cb_password")
        
    def add(self, dataset_id, load=False, collection=None):
        """
        Add the experiment to the list of experiments and save results to files.
        
        Args:
            dataset_id: Identifier for the dataset used in the experiment
            load: Boolean flag to determine if results should be loaded to Couchbase
            collection: Optional custom Couchbase collection name
        """
        # Create a directory with the name Experiment_<Experiment_id>
        experiment_dir = f"Experiment_{self.options.experiment_id}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save detailed results (all except first item) to JSON and CSV
        with open(os.path.join(experiment_dir, "output.json"), "w") as f:
            json.dump(self.output[1:], f, indent=4)
        with open(os.path.join(experiment_dir, "output.csv"), "w") as f:
            df = pd.json_normalize(self.output[1:])
            df.to_csv(f, index=False)
            
        # Save averaged results (first item only) to JSON and CSV
        with open(os.path.join(experiment_dir, "averaged_output.json"), "w") as f:
            json.dump(self.output[:1], f, indent=4)
        with open(os.path.join(experiment_dir, "averaged_output.csv"), "w") as f:
            df = pd.json_normalize(self.output[:1])
            df.to_csv(f, index=False)
        
        # Create the experiment metadata with configuration and results summary
        self.metadata= {
            "experiment_id": self.options.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "chunk_size": self.options.chunk_size,
            "chunk_overlap": self.options.chunk_overlap,
            "embedding_model": self.options.embedding_model,
            "embedding_dimension": self.options.embedding_dimension,
            "llm_model": self.options.llm_model,
            "metrics": self.metrics,
            "dataset_size": len(self.output)-1,
            "dataset_id": dataset_id
        }
        
        # Save metadata to a separate file
        with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
            
        # Optionally load data to Couchbase if requested
        if load:
            self.load_to_couchbase(collection)
        
    def load_to_couchbase(self,  _collection):
        """
        Load the experiment data to Couchbase database.
        
        Args:
            _collection: Optional custom collection name to use instead of default
        """
        # If a separate collection is defined by the user, use that collection
        if _collection is not None:
            self.collection = _collection
            
        object_id = 1
        cb = self.connect()
        cb_coll = cb.scope(self.scope).collection(self.collection)
        batch_exceptions = []
        batch_size = 100  # Process data in batches of 100 for efficiency

        # Create Couchbase object for the averaged output and load it
        data = dict()
        averaged_data = self.output[:1]
        averaged_data[0]["metadata"] = self.metadata  # Attach metadata to the averaged result
        data[f"{self.options.experiment_id}_average"] = averaged_data[0]
        results = cb_coll.upsert_multi(data)
        
        # Track any exceptions during the averaged data upload
        if len(results.exceptions) > 0:
            batch_exceptions.append(results.exceptions)
        
        # Create Couchbase objects for the individual data points and load them in batches
        for i in range(0, len(self.output[1:]), batch_size):
            data = dict()
            for object in self.output[1:][i:i+batch_size]:
                object["metadata"] = self.metadata  # Attach metadata to each individual result
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
        
            
    def connect(self):
        """
        Establish connection to Couchbase cluster and ensure required scopes and collections exist.
        
        Returns:
            Couchbase bucket reference if connection successful, None otherwise
        """
        # Check if the environment variable `has_cert_file` field is set
        has_cert_file = os.getenv("has_cert_file") or False
            
        # Configure authentication based on certificate availability
        if has_cert_file:
            auth = PasswordAuthenticator(self.username, self.password, cert_path="/root/cert.txt")
        else:
            auth = PasswordAuthenticator(self.username, self.password)
            
        # Set up connection options with WAN development profile
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")

        # Determine environment and connect accordingly
        env = os.getenv("env") or "dev"
        try:
            if env == "dev":
                # Skip TLS verification in development environment
                self.cluster = Cluster.connect(self.cluster_url, options, tls_verify=TLSVerifyMode.NONE)
            else:
                # Use default TLS verification in production
                self.cluster = Cluster.connect(self.cluster_url, options)
            # Wait for cluster to be ready with timeout
            self.cluster.wait_until_ready(timedelta(seconds=30))
            
        except CouchbaseException as e:
            print(f"Unable to connect to the cluster: {e}")
            return None
        
        # Get bucket reference
        cb = self.cluster.bucket(self.bucket)

        # Get the bucket manager for collection operations
        bucket_manager = cb.collections()

        # Create a new scope if it does not exist
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(s.name == self.scope for s in scopes)
        if not scope_exists:
            bucket_manager.create_scope(self.scope)

        # Create a new collection within the scope if collection does not exist
        if scope_exists:
            # Get all collections in the target scope
            collections = [c.name for s in scopes if s.name == self.scope for c in s.collections]
            collection_exists = self.collection in collections
            if not collection_exists:
                bucket_manager.create_collection(self.scope, self.collection)
        else:
            # If scope was just created, create the collection as well
            bucket_manager.create_collection(self.scope, self.collection)
            
        return cb