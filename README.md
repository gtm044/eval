# RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using RAGAS and synthesizing ground truth data from raw documents.

## Overview

This framework provides tools and metrics to evaluate RAG systems across three key components:
- Chunking evaluation
- Retrieval evaluation
- Generation evaluation

The framework integrates with RAGAS, a popular RAG evaluation library, and provides a structured approach for experiment management and result persistence.

## Features

### 1. Evaluation Dataset Management
- Structured data class (`EvalDataset`) for managing evaluation datasets
- Support for questions, answers, responses, and contexts (both reference and retrieved)
- Automatic validation of dataset integrity
- JSON conversion utilities

### 2. Comprehensive Metrics

#### Chunking Metrics
- Average chunk size evaluation (normalized index from -inf to 1.0, higher is better, >0.5 is acceptable)
- Will be the same for all data points, indicates the avg_chunk_size for the entire ground truth dataset.

#### Retrieval Metrics
- Context precision
- Context recall
- Semantic similarity analysis

#### Generation Metrics
- Answer relevancy
- Faithfulness evaluation
- Answer correctness

### 3. Experiment Management
- Structured experiment tracking with unique experiment IDs
- Metadata management including chunking, embedding, and LLM parameters
- Result persistence with Couchbase integration
- Flexible experiment configuration via `ExperimentOptions`

### 4. Data Generation
- Synthetic question-answer generation
- Document processing capabilities
- Support for multiple input formats

## Installation

1. Clone the repository
2. Install the package:
```bash
cd eval
pip install -e .
python -m spacy download en_core_web_sm
```

## Usage

### Basic Evaluation

```python
from src.data.dataset import EvalDataset
from src.evaluator.validation import ValidationEngine

# Create dataset
dataset = EvalDataset(
    questions=["What is RAG?"],
    answers=[["RAG is a retrieval-augmented generation system"]],
    responses=["RAG combines retrieval with generation"],
    reference_contexts=["RAG systems use retrieval to enhance generation"],
    retrieved_contexts=[["RAG: retrieval-augmented generation"]]
)

# Run evaluation with RAGAS metrics
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness

engine = ValidationEngine(
    dataset=dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size]
)
results = engine.evaluate()
```
Results are stored in folder named `.results`

### Experiment Management

```python
from src.controller.options import ExperimentOptions
from src.controller.manager import Experiment

# Configure experiment
experiment_options = ExperimentOptions(
    experiment_id="exp_001",
    dataset_id="dataset_001",
    metrics=["context_precision", "context_recall", "faithfulness"],
    chunk_size=100,
    chunk_overlap=20,
    embedding_model="text-embedding-3-large",
    embedding_dimension=3072,
    llm_model="gpt-4"
)

# Create experiment and store results
experiment = Experiment(dataset=dataset, options=experiment_options)

# Load the experiment config and results to couchbase kv cluster
experiment.load_to_couchbase()

# Retrieve experiment results from the couchbase kv store
experiment_result = Experiment().retrieve("exp_001")
```
Results are stored in `.results-<experiment_id>`

## Configuration

The framework uses environment variables for Couchbase configuration:

```env
bucket=your_bucket
scope=your_scope
collection=your_collection
cluster_url=your_cluster_url
cb_username=your_username
cb_password=your_password
```

## Roadmap

1. **Advanced Metrics Integration**
   - Custom metric development
   - Multi-turn conversation evaluation
   - Hallucination detection

2. **Multimodal RAG Support**
   - Image retrieval evaluation
   - Table content processing
   - Cross-modal metrics

3. **Report Generation**
   - LLM-based analysis
   - Structured reporting schema
   - Interactive dashboards