# Create a experiment management system that manages evalautions as experiments
import os
import json
from typing import List
from numpy import object_
import pandas as pd
from datetime import datetime, timedelta
from src.controller.options import ExperimentOptions
from src.evaluator.options import ValidationOptions
from src.evaluator.validation import ValidationEngine
from src.data.load import LoadOperator

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, TLSVerifyMode
from couchbase.exceptions import CouchbaseException, DocumentNotFoundException
from couchbase.kv_range_scan import PrefixScan

class Experiment:
    def __init__(self, options: ExperimentOptions, evaluation_output: List[dict]):
        self.options = options
        self.output, self.metrics = evaluation_output
        # Couchbase connection credentials
        self.bucket = os.getenv("bucket")
        self.scope = os.getenv("scope")
        self.collection = os.getenv("collection")
        self.cluster_url = os.getenv("cluster_url")
        self.username = os.getenv("cb_username")
        self.password = os.getenv("cb_password")
        
    def add(self, dataset_id, load=False, collection=None):
        """
        Add the experiment to the list of experiments
        """
        # create a directory with the the name Experiment_<Experiment_id>
        experiment_dir = f"Experiment_{self.options.experiment_id}"
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, "output.json"), "w") as f:
            json.dump(self.output[1:], f, indent=4)
        with open(os.path.join(experiment_dir, "averaged_output.json"), "w") as f:
            json.dump(self.output[:1], f, indent=4)
        with open(os.path.join(experiment_dir, "output.csv"), "w") as f:
            df = pd.json_normalize(self.output[1:])
            df.to_csv(f, index=False)
        with open(os.path.join(experiment_dir, "averaged_output.csv"), "w") as f:
            df = pd.json_normalize(self.output[:1])
            df.to_csv(f, index=False)
        
        # Create the experiment metadata
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
        
        with open(os.path.join(experiment_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
            
        if load:
            self.load_to_couchbase(experiment_dir, collection)
        
    def load_to_couchbase(self, experiment_dir, _collection):
        """
        Load the experiment data to Couchbase
        """
        # If a seperate collection is defined by the user, use that collection
        if _collection is not None:
            self.collection = _collection
            
        object_id = 1
        cb = self.connect()
        cb_coll = cb.scope(self.scope).collection(self.collection)
        batch_exceptions = []
        batch_size = 100

        # Create couchbase object for the averaged output and load it
        data = dict()
        averaged_data = self.output[:1]
        averaged_data["metadata"] = self.metadata
        data[f"{self.options.experiment_id}_average"] = averaged_data
        results = cb_coll.upsert_multi(data)
        
        if len(results.exceptions) > 0:
            batch_exceptions.append(results.exceptions)
        
        # Create couchbase objects for the individual data points and load them
        for i in range(0, len(self.output[1:]), batch_size):
            data = dict()
            for object in self.output[1:][i:i+batch_size]:
                object["metadata"] = self.metadata
                data[f"{self.options.experiment_id}_{object_id}"] = object
                object_id += 1
            results = cb_coll.upsert_multi(data)
            if len(results.exceptions) > 0:
                batch_exceptions.append(results.exceptions)
        
        if len(batch_exceptions) > 0:
            raise Exception(batch_exceptions)
        else:
            print(f"Experiment data loaded successfully with experiment id: {self.options.experiment_id}")
        
            
    def connect(self):
        # Check if the environment variable `has_cert_file` field
        has_cert_file = os.getenv("has_cert_file") or False
            
        # Connect to the Couchbase cluster
        if has_cert_file:
            auth = PasswordAuthenticator(self.username, self.password, cert_path="/root/cert.txt")
        else:
            auth = PasswordAuthenticator(self.username, self.password)
            
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")

        env = os.getenv("env") or "dev"
        try:
            if env == "dev":
                self.cluster = Cluster.connect(self.cluster_url, options, tls_verify=TLSVerifyMode.NONE)
            else:
                self.cluster = Cluster.connect(self.cluster_url, options)
            self.cluster.wait_until_ready(timedelta(seconds=30))
            
        except CouchbaseException as e:
            print(f"Unable to connect to the cluster: {e}")
            return None
        
        # Get bucket ref
        cb = self.cluster.bucket(self.bucket)

        # Get the bucket manager
        bucket_manager = cb.collections()

        # Create a new scope if it does not exist
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(s.name == self.scope for s in scopes)
        if not scope_exists:
            bucket_manager.create_scope(self.scope)

        # Create a new collection within the scope if collection does not exist
        if scope_exists:
            collections = [c.name for s in scopes if s.name == self.scope for c in s.collections]
            collection_exists = self.collection in collections
            if not collection_exists:
                bucket_manager.create_collection(self.scope, self.collection)
        else:
            bucket_manager.create_collection(self.scope, self.collection)
            
        return cb
            