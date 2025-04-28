import json
import pandas as pd
from typing import List, Dict, Set, Optional
from collections import defaultdict
from src.data.cluster.utils import doc_to_nl, text_to_relations, text_to_relations_llm, get_default_save_directory
from sentence_transformers import SentenceTransformer
import hdbscan
import json
import os

class SemanticCluster:
    def __init__(self, texts: Optional[List[str]] = None):
        if texts:
            self.documents = texts
        else:
            self.documents = None
        self.document_knowledge_map = {} # Mapping documents to their knowledge(entity, relation, entity)
        self.document_index = {}
        
    def process_json(self, json_path: str, field: str = None, limit: Optional[int] = None):
        with open(json_path, 'r') as f:
            self.documents = json.load(f)
        if field:
            self.documents = [{field: doc[field]} for doc in self.documents]
        if limit:
            self.documents = self.documents[:limit]
                
    def process_csv(self, csv_path: str, field: str = None, limit: Optional[int] = None):
        df = pd.read_csv(csv_path)
        if field:
            df = df[field]
        if limit:
            df = df.head(limit)
        self.documents = df.to_dict(orient='records')
        
    def build_clusters(self, **kwargs):
        # Map documents to an index, which is used to map the relations to the documents.
        for i, document in enumerate(self.documents):
            self.document_index[f"doc_{i}"] = document     
        self.clean_documents(**kwargs)
        self.extract_relations()
        self.embed_and_cluster_documents()
        
    # Find a method to extract entity-relation triads from the json documents.
    def extract_relations(self):
        nl_texts = doc_to_nl(self.documents)
        # Extract relations using the rebel model/llm
        # raw_relations = text_to_relations_llm(nl_texts)
        raw_relations = text_to_relations(nl_texts)
        for i, relations in enumerate(raw_relations):
            self.document_knowledge_map[f"doc_{i}"] = relations
        
    def embed_and_cluster_documents(self, use_hdbscan: bool = True, num_clusters: int = 5, min_cluster_size: int = 2):
        print("Clustering documents using knowledge embeddings...")

        # 1. concat the triples of each document to form a text
        doc_texts = []
        doc_ids = []
        for doc_id, triples in self.document_knowledge_map.items():
            text = ". ".join([f"{t['head']} {t['type']} {t['tail']}" for t in triples])
            doc_texts.append(text)
            doc_ids.append(doc_id)

        if not doc_texts:
            raise ValueError("No relations found. Ensure `build_graph()` is run first.")

        # 2. Embed texts
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(doc_texts)

        # 3. Cluster embeddings
        if use_hdbscan:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(embeddings)
        else:
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            labels = clusterer.fit_predict(embeddings)

        # 4. Build cluster mapping
        clustered_docs = defaultdict(list)
        for label, doc_id in zip(labels, doc_ids):
            clustered_docs[label].append(doc_id)

        self.document_clusters = clustered_docs

        # print(f"\nCluster Results from {algo_name}:")
        # for cluster_id, docs in clustered_docs.items():
        #     if cluster_id == -1:
        #         print(f"Noise/Outliers: {docs}")
        #     else:
        #         print(f"Cluster {cluster_id}: {docs}")

        directory = get_default_save_directory()
        cluster_documents = {}
        for cluster_id, doc_ids in clustered_docs.items():
            if cluster_id == -1:
                continue
            cluster_documents[cluster_id] = [self.document_index[doc_id] for doc_id in doc_ids]
        
        for cluster_id, documents in cluster_documents.items():
            filename = f"{directory}/cluster_{cluster_id}.json"
            with open(filename, "w") as f:
                json.dump(documents, f, indent=4)
                
        return clustered_docs
        
    def clean_documents(self, **kwargs):
        # Flatten nested dict fields
        flattened_documents = []
        for document in self.documents:
            out = {}
            def flatten(x, name=''):
                if type(x) is dict:
                    for a in x:
                        flatten(x[a], f"{name}{a}_")
                elif type(x) is list:
                    for i, a in enumerate(x):
                        flatten(a, f"{name}{i}_")
                else:
                    out[name[:-1]] = x
            flatten(document)
            flattened_documents.append(out)
        self.documents = flattened_documents
        
        # Produce an error if there are null values in the json
        for document in self.documents:
            if any(value is None for value in document.values()):
                if kwargs.get('clean_documents', True):
                    self.documents = [doc for doc in self.documents if not any(value is None for value in doc.values())]
                    if len(self.documents) == 0:
                        raise ValueError("Too many null values found in the dataset, more than the feasible limit. Please clean the json/csv before building the graph for optimal results.")
                else:
                    raise ValueError("Null values found in the dataset, please clean the json before building the graph for optimal results. Provide an argument 'clean_documents=True' to the builder function to remove the null values. Continuing with building the graph with null values.")
        
        # Check for duplicate documents
        seen = set()
        duplicate_documents = []
        for document in self.documents:
            sorted_doc = tuple(sorted(document.items()))
            if sorted_doc in seen:
                duplicate_documents.append(document)
            else:
                seen.add(sorted_doc)      
        if duplicate_documents:
                self.documents = [doc for doc in self.documents if doc not in duplicate_documents]
                
        
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build semantic clusters from documents')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV or JSON file')
    parser.add_argument('--input_type', type=str, choices=['csv', 'json'], default='csv',
                        help='Type of input file (csv or json)')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of documents to process')
    parser.add_argument('--cluster_type', type=str, choices=['hdbscan', 'kmeans'], default='hdbscan', 
                        help='Clustering algorithm to use (hdbscan or kmeans)')
    parser.add_argument('--num_clusters', type=int, default=5, help='Number of clusters for KMeans')
    parser.add_argument('--min_cluster_size', type=int, default=2, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--field', type=str, default=None, help='Field to use for JSON input')
    
    args = parser.parse_args()
    
    kg = SemanticCluster()
    
    if args.input_type == 'csv':
        kg.process_csv(args.input, field=args.field, limit=args.limit)
    else:  # json
        kg.process_json(args.input, field=args.field, limit=args.limit)
    
    use_hdbscan = args.cluster_type == 'hdbscan'
    clusters = kg.build_clusters(use_hdbscan=use_hdbscan, 
                                num_clusters=args.num_clusters,
                                min_cluster_size=args.min_cluster_size)
    
    os.makedirs("clusters", exist_ok=True)
    
    cluster_documents = {}
    for cluster_id, doc_ids in clusters.items():
        if cluster_id == -1:
            continue
        cluster_documents[cluster_id] = [kg.document_index[doc_id] for doc_id in doc_ids]
    
    for cluster_id, documents in cluster_documents.items():
        filename = f"clusters/cluster_{cluster_id}.json"
        with open(filename, "w") as f:
            json.dump(documents, f, indent=4)
        
    print(f"Saved {len(cluster_documents)} clusters to the 'clusters' directory")