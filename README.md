# RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems and synthesizing ground truth data.

## Overview

This framework provides tools and metrics to evaluate RAG systems across three key components:
- Chunking evaluation
- Retrieval evaluation
- Generation evaluation

## Features

### 1. Evaluation Dataset Management
- Structured data class for managing evaluation datasets
- Support for questions, answers, responses, and contexts
- Automatic validation of dataset integrity
- JSON conversion utilities

### 2. Comprehensive Metrics

#### Chunking Metrics
- Average chunk size evaluation
- Normalized scoring system
- Configurable chunk size limits

#### Retrieval Metrics
- Context relevance scoring
- Embedding similarity analysis
- Named entity matching
- Retrieval accuracy

#### Generation Metrics
- BLEU score
- ROUGE score
- Faithfulness evaluation
- Response similarity analysis

### 3. Experiment Management
- Structured experiment tracking
- Metadata management
- Result persistence
- Couchbase integration for storage

### 4. Data Generation
- Synthetic question-answer generation
- Document expansion capabilities
- Support for multiple input formats

## Installation

1. Clone the repository
2. Install the package:
```bash
pip install e .
python -m spacy download en_core_web_sm
```

## Usage

### Basic Evaluation

```python
from src.data.dataset import EvalDataset
from src.evaluator.validation import ValidationEngine
from src.evaluator.options import ValidationOptions

# Create dataset
dataset = EvalDataset(
    questions=["What is RAG?"],
    answers=["RAG is a retrieval-augmented generation system"],
    responses=["RAG combines retrieval with generation"],
    reference_contexts=["RAG systems use retrieval to enhance generation"],
    retrieved_contexts=["RAG: retrieval-augmented generation"]
)

# Configure evaluation
options = ValidationOptions(
    metrics=["bleu_score", "rouge_score", "faithfulness"],
    generateReport=True
)

# Run evaluation
engine = ValidationEngine(dataset=dataset, options=options)
results = engine.evaluate()
```

### Experiment Management

```python
from src.controller.options import ExperimentOptions
from src.controller.manager import Experiment

# Configure experiment
experiment_options = ExperimentOptions(
    experiment_id="exp_001",
    chunk_size=100,
    chunk_overlap=20,
    embedding_model="text-embedding-3-large",
    embedding_dimension=3072,
    llm_model="gpt-4"
)

# Create and save experiment
experiment = Experiment(experiment_options, evaluation_results)
experiment.add(dataset_id="dataset_001", load=True)
```

## Configuration

The framework uses environment variables for configuration:

```env
bucket=your_bucket
scope=your_scope
collection=your_collection
cluster_url=your_cluster_url
cb_username=your_username
cb_password=your_password
```

## Output Format

The evaluation produces detailed metrics in both JSON and CSV formats:

```json
{
    "avg_chunk_size": -127.667,
    "retrieval_accuracy": 0.0,
    "avg_context_score": [0.353, 0.365],
    "avg_embedding_similarity": 0.241,
    "avg_named_entity_score": 0.007,
    "avg_bleu_score": 0.643,
    "avg_rouge_score": [0.448, 0.61],
    "avg_faithfulness": 7.333,
    "avg_response_similarity": 0.883
}
```

## Roadmap

1. **Data Expansion Methods**
   - Text AutoAugment integration
   - Embedding-based noise injection

2. **Multimodal RAG Support**
   - Image retrieval evaluation
   - Table content processing
   - Cross-modal metrics

3. **Report Generation**
   - LLM-based analysis
   - Structured reporting schema
   - Visualization components