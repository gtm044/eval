# src/data/load.py

from typing import List, Optional
from datetime import timedelta
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid
import json

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, TLSVerifyMode
from couchbase.exceptions import CouchbaseException, DocumentNotFoundException
from couchbase.kv_range_scan import PrefixScan

from dataset import EvalDataset

load_dotenv()    
    
class LoadOperator:
    """
    Load Operator to push processed document into a keyspace in Couchbase.

    This operator takes a keyspace and authentication details from env variables to connect to a
    Couchbase cluster and inserts it into the keyspace.
    
    Returns the dataset id for the loaded documents.
    """
    def __init__(self, data:Optional[EvalDataset] = None, dataset_description:Optional[str] = None):
        if not data:
            pass
        self.bucket = os.getenv("bucket")
        self.scope = os.getenv("scope")
        self.collection = os.getenv("collection")
        self.cluster_url = os.getenv("cluster_url")
        self.username = os.getenv("cb_username")
        self.password = os.getenv("cb_password")
        
        self.data = data
        self.doc_id = 1
        self.dataset_id = self.generate_parent_id()
        self.dataset_description = dataset_description
        
    def generate_parent_id(self):
        """
        Generates an id for the entire dataset.
        Used to group all the documents belonging to a dataset in the collection.
        """
        return str(uuid.uuid4())
    
    def load_docs(self):
        """
        Load the documents into the Couchbase collection.
        """
        cb = self.connect()
        cb_coll = cb.scope(self.scope).collection(self.collection)
        json_data = self.data.to_json() # Return a list of dictionaries
        batch_size = 100
        batch_exceptions = []
        
        for i in range(0, len(json_data), batch_size):
            docs_to_insert = dict()
            for doc in json_data[i:i+batch_size]:
                print(doc)
                doc["doc_id"] = self.doc_id
                self.doc_id += 1
                doc["meta-data"] = {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_description": self.dataset_description if self.dataset_description else "",
                    "dataset_id": self.dataset_id
                }
                key = f"{self.dataset_id}_{doc['doc_id']}"
                docs_to_insert[key] = doc
                
            # Insert the documents into the collection in batches of 100
            results = cb_coll.upsert_multi(docs_to_insert)
            exceptions = results.exceptions
            if len(exceptions) > 0:
                batch_exceptions.append(exceptions)
        
        if len(batch_exceptions) > 0:
            raise Exception(batch_exceptions)
        else:
            print(f"Documents loaded successfully with dataset id: {self.dataset_id}")
            
            
    def retrieve_docs(self, dataset_id: str, doc_id: Optional[List[int]] = None):
        """
        Retrieve the documents from the collection and return an EvalDataset object.

        - Fetches all documents with the dataset id if `doc_id` is not provided.
        - Fetches specific documents if `doc_id` is provided.
        """
        cb = self.connect()
        cb_coll = cb.scope(self.scope).collection(self.collection)
        content = []
        if doc_id is None:
            prefix = f"{dataset_id}_"
            result = cb_coll.scan(PrefixScan(prefix))
            for r in result:
                content.apend(r.content_as[dict])
        else:
            keys = [f"{dataset_id}_{i}" for i in doc_id]
            result = cb_coll.get_multi(keys)
            for r in result.results.values():
                content.append(r.value)
        
        ## Create the EvalDataset object with only those fields that has a value not None
        output_dict = {}
        if content[0]["question"]:
            output_dict["questions"] = [c["question"] for c in content]
        if content[0]["answer"]:
            output_dict["answers"] = [c["answer"] for c in content]
        if content[0]["response"]:
            output_dict["responses"] = [c["response"] for c in content]
        if content[0]["reference_context"]:
            output_dict["reference_contexts"] = [c["reference_context"] for c in content]
        if content[0]["retrieved_context"]:    
            output_dict["retrieved_contexts"] = [c["retrieved_context"] for c in content]
        
        
        return EvalDataset(**output_dict)
        
        

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

    
    
if __name__ == '__main__':
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": ["Paris", "Joe Biden"],
        "responses": ["Paris", "Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"],
        "retrieved_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"]
    }
    dataset = EvalDataset(**data)
    dataset.to_json("test.json")
    loader = LoadOperator(data=dataset, dataset_description="Test dataset")
    loader.load_docs()
    print(loader.dataset_id)
    # Instantiate a new loader to retrieve the documents
    retriever = LoadOperator() # No arguments for the retriever load operator
    print(retriever.retrieve_docs(loader.dataset_id, [1, 2]))