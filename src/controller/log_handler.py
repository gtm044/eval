## Python code to retrieve the agentc logs for the corresponsding 

# Given session id, couchbase cluster url, username, password bucket name, scope name, collection name, retrieve the logs with the corresponding session id, order by "timestamp" in ascending order

import couchbase
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, TLSVerifyMode
from couchbase.exceptions import CouchbaseException
from couchbase.kv_range_scan import PrefixScan
import json
from src.data.load import LoadOperator
import os
from dotenv import load_dotenv

load_dotenv()

class LogHandler:
    def __init__(self, cluster_url, username, password, bucket_name, scope_name, collection_name):
        self.cluster_url = cluster_url
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self.scope_name = scope_name
        self.collection_name = collection_name
        
    def retrieve_logs(self, session_id, catalog_folder=None):
        
        if catalog_folder:
            # The catalog folder contains a file called "llm-activity.log" -> contains all the logs (analogous to the couchbase collection)
            # File contains a set of dicts, each representing a log entry
            with open(os.path.join(catalog_folder, "llm-activity.log"), "r") as f:
                logs = f.readlines()
                
            # Parse the logs into a list of dicts
            logs = [json.loads(log) for log in logs]
            
            # Filter the logs for the corresponding session id
            logs = [log for log in logs if log["session"] == session_id]
            
            # Remove duplicate rows ignoring the field 'timestamp'
            seen = set()
            unique_data = []
            for item in logs:
                key = frozenset((k, str(v)) for k, v in item.items() if k != 'timestamp')
                if key not in seen: 
                    seen.add(key)
                    unique_data.append(item)
                    
            # Order the unique_data by timestamp in ascending order
            unique_data = sorted(unique_data, key=lambda x: x['timestamp'])
            
            # Dump 
            with open(f"logs2_{session_id}.json", "w") as f:
                json.dump(unique_data, f, indent=4)
                
            print(len(unique_data))
            
            # Print the number of documents with the content.kwargs.content with the value "What type of room is offered in the \"Clean & quiet apt home by the park\"?"
            count = 0
            for item in unique_data:
                for key, value in item.items():
                    if key == "content" and "kwargs" in value and "content" in value["kwargs"]:
                        if value["kwargs"]["content"] == "What type of room is offered in the \"Clean & quiet apt home by the park\"?":
                            count += 1
                            
            print(count)
            
            
            return unique_data
        
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
        
        # _ = self.cluster.query(f"CREATE PRIMARY INDEX ON `{self.bucket_name}`")
        # index_query = f"CREATE INDEX timestamp_index ON `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}`(timestamp)"
        # _ = self.cluster.query(index_query)
        
        query = f"SELECT a.* FROM `{self.bucket_name}`.`{self.scope_name}`.`{self.collection_name}` as a WHERE a.session = '{session_id}'"
        result = self.cluster.query(query)
        
        json_list = []
        for row in result.rows():
            json_list.append(row)
            
        # Remove duplicate rows ignoring the field 'timestamp'
        seen = set()
        unique_data = []
        for item in json_list:
            # Exclude the timestamp field since we want to ignore it for deduplication
            key = frozenset((k, str(v)) for k, v in item.items() if k != 'timestamp')
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        # Order the unique_data by timestamp in ascending order
        unique_data = sorted(unique_data, key=lambda x: x['timestamp'])
        
        # Dump 
        with open(f"logs_{session_id}.json", "w") as f:
            json.dump(unique_data, f, indent=4)
            
        print(len(unique_data))
        return unique_data
    
    
if __name__ == "__main__":
    log_handler = LogHandler(cluster_url="couchbase://localhost", username="Administrator", password="password", bucket_name="travel-sample", scope_name="agent_activity", collection_name="raw_logs")
    log_handler.retrieve_logs("doc2", catalog_folder = "/Users/goutham.krishnan/Documents/Work/cwd_eval/chat-with-data/.agent-activity")