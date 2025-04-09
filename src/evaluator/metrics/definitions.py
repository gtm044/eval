# Byepasses ragas metric definitions to unify metrics.

from ragas.metrics.base import Metric
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness, answer_similarity
from src.evaluator.metrics.chunking import avg_chunk_size
from src.evaluator.metrics.retrieval import context_similarity, context_score, named_entity_score
from src.evaluator.metrics.generation import llm_grading

faithfulness = faithfulness
answer_relevancy = answer_relevancy
answer_correctness = answer_correctness
context_precision = context_precision
context_recall = context_recall
answer_similarity = answer_similarity
avg_chunk_size = avg_chunk_size
context_similarity = context_similarity
context_score = context_score
named_entity_score = named_entity_score
llm_grading = llm_grading