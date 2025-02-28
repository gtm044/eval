# src/evaluator/validation.py
from src.evaluator.metrics import chunking, generation, retrieval
from src.evaluator.options import ValidationOptions
from src.data.dataset import EvalDataset

class ValidationEngine:
    def __init__(
        self,
        dataset: EvalDataset,
        options: ValidationOptions,
    ):
        self.dataset = dataset
        self.options = options
        
    def evaluate(self):
        """
        Evaluate the performance of the RAG model.
        """
        # Calculate the metrics
        output = []
        list_of_metrics = []
        
        # If segments are provided, calculate the metrics for the segments
        if self.options.segments:
            for segment in self.options.segments:
                if segment == "chunking":
                    list_of_metrics.append("avg_chunk_size")
                elif segment == "retrieval":
                    list_of_metrics.extend(["context_score", "embedding_similarity", "named_entity_score", "retrieval_accuracy"])
                elif segment == "generation":
                    list_of_metrics.extend(["bleu_score", "rouge_score", "faithfulness", "response_similarity"])
        else:
            list_of_metrics = self.options.metrics
            
        metrics = self.calculate_metrics(list_of_metrics)
    
        # Create a list of dictionaries containing the data points and the corresponsing evaluation metrics
        # Initially add a dictionary for the average metrics (avg_chunk_size, retrieval_accuracy)
        output.append({
            "avg_chunk_size": round(metrics["avg_chunk_size"], 2),
            "retrieval_accuracy": round(metrics["retrieval_accuracy"], 2),
            "avg_context_score": [sum(x) / len(x) for x in zip(*metrics["context_score"])],
            "avg_embedding_similarity": round(sum(metrics["embedding_similarity"]) / len(metrics["embedding_similarity"]), 2),
            "avg_named_entity_score": round(sum(metrics["named_entity_score"]) / len(metrics["named_entity_score"]), 2),
            "avg_bleu_score": round(sum(metrics["bleu_score"]) / len(metrics["bleu_score"]), 2),
            "avg_rouge_score": [sum(x) / len(x) for x in zip(*metrics["rouge_score"])],
            "avg_faithfulness": round(sum(metrics["faithfulness"]) / len(metrics["faithfulness"]), 2),
            "avg_response_similarity": round(sum(metrics["response_similarity"]) / len(metrics["response_similarity"]), 2)
        })
        
        for i in range(len(self.dataset.questions)):
            data = {
                "question": self.dataset.questions[i],
                "answer": self.dataset.answers[i],
                "response": self.dataset.responses[i],
                "reference_context": self.dataset.reference_contexts[i],
                "retrieved_context": self.dataset.retrieved_contexts[i]
            }
            for key, value in metrics.items():
                # Handle metrics with just one single unified value
                if key=="avg_chunk_size" or key=="retrieval_accuracy":
                    continue   
                else:
                    data[key] = value[i]
            output.append(data)
              
        return output, list_of_metrics
    
    def calculate_metrics(self, metrics):
        scores = {}
        for metric in metrics:
            if metric == "avg_chunk_size":
                scores["avg_chunk_size"] = chunking.avg_chunk_size(self.dataset.reference_contexts)
            elif metric == "jaccard_index":
                pass # Not going to implement this as of now, need to figure out how the chunk ground truth is generated.
            elif metric == "context_score":
                scores["context_score"] = retrieval.context_score(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
            elif metric == "embedding_similarity":
                scores["embedding_similarity"] = retrieval.embedding_similarity(self.dataset.questions, self.dataset.retrieved_contexts)
            elif metric == "named_entity_score":
                scores["named_entity_score"] = retrieval.named_entity_score(self.dataset.questions, self.dataset.retrieved_contexts)
            elif metric == "retrieval_accuracy":
                scores["retrieval_accuracy"] = retrieval.retrieval_accuracy(self.dataset.reference_contexts, self.dataset.retrieved_contexts)
            elif metric == "bleu_score":
                scores["bleu_score"] = generation.bleu_score(self.dataset.answers, self.dataset.responses)
            elif metric == "rouge_score":
                scores["rouge_score"] = generation.rouge_score(self.dataset.answers, self.dataset.responses)
            elif metric == "faithfulness":
                scores["faithfulness"] = generation.faithfulness(self.dataset.retrieved_contexts, self.dataset.responses)
            elif metric == "response_similarity":
                scores["response_similarity"] = generation.response_similarity(self.dataset.answers, self.dataset.responses)
            else:
                raise ValueError(f"Metric {metric} is not implemented yet.")
        return scores
    
    
    # Not completed yet
    def calculate_index(self, avg_scores):
        """
        Calculate the index based on the weighted average of the metrics.
        
        Args:
            avg_scores (dict): Dictionary containing the average scores for each metric
            
        Returns:
            dict: Dictionary containing the calculated indices for chunking, retrieval, and generation
        """
        # Define weights for different components
        weight = 0.5
        
        # Calculate chunking index
        chunking_index = 0
        if "avg_chunk_size" in avg_scores and "retrieval_accuracy" in avg_scores:
            chunking_index = (weight * avg_scores["avg_chunk_size"] + (1 - weight) * avg_scores["retrieval_accuracy"]) / 2
        
        # Calculate retrieval index
        retrieval_metrics = ["context_score", "embedding_similarity", "named_entity_score", "retrieval_accuracy"]
        retrieval_values = []
        for metric in retrieval_metrics:
            if metric in avg_scores:
                retrieval_values.append(weight * avg_scores[metric])
        
        retrieval_index = sum(retrieval_values) / len(retrieval_values) if retrieval_values else 0
        
        # Calculate generation index
        generation_metrics = ["bleu_score", "rouge_score", "faithfulness", "response_similarity"]
        generation_values = []
        for metric in generation_metrics:
            if metric in avg_scores:
                generation_values.append(weight * avg_scores[metric])
        
        generation_index = sum(generation_values) / len(generation_values) if generation_values else 0
        
        # Calculate overall RAG index
        rag_index = (chunking_index + retrieval_index + generation_index) / 3
        
        return {
            "chunking_index": chunking_index,
            "retrieval_index": retrieval_index,
            "generation_index": generation_index,
            "rag_index": rag_index
        }
        
        
if __name__=='__main__':
    data = {
        "questions": ["What is the capital of France?", "Who is the president of the USA?"],
        "answers": ["Paris is the capital of france", "Joe Biden is the president of the USA"],
        "responses": ["Capital of france is Paris", "President of the USA is Joe Biden"],
        "reference_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"],
        "retrieved_contexts": ["Paris is the capital of France", "Joe Biden is the president of the USA"]
    }
    _dataset = EvalDataset(**data)
    _options = ValidationOptions(
        experiment_id="1234",
        metrics = [
            "avg_chunk_size", "context_score", "embedding_similarity", "named_entity_score", "retrieval_accuracy", "bleu_score", "rouge_score", "faithfulness", "response_similarity"
        ],
        generateReport=False
    )
    eval = ValidationEngine(dataset=_dataset, options=_options, output_dir="output_data/")
    result = eval.evaluate()