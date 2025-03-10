# RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using RAGAS and synthesizing ground truth data from raw documents.

## Overview

This framework provides tools and metrics to evaluate RAG systems across three key components:
- Chunking evaluation
- Retrieval evaluation
- Generation evaluation

The framework integrates with RAGAS, a popular RAG evaluation library, and provides a structured approach for experiment management, storage and result persistence.

## Features

### 1. Evaluation Dataset Management
- Structured data class ([`EvalDataset`](src/data/dataset.py)) for managing evaluation datasets
- Support for questions, answers, responses, and contexts (both reference and retrieved)
- JSON serialization/deserialization.

### 2. Comprehensive Metrics

#### Chunking Metrics
- Average chunk size evaluation (normalized index from -inf to 1.0, higher is better, >0.5 is acceptable)
- Will be the same for all data points, indicates the avg_chunk_size for the entire ground truth dataset.

#### Retrieval Metrics
- Context precision
- Context recall

#### Generation Metrics
- Answer relevancy
- Faithfulness evaluation
- Answer correctness

### 3. Experiment Management
- Structured experiment tracking with unique experiment IDs
- Metadata management including chunking, embedding, and LLM parameters
- Result persistence with Couchbase integration
- Flexible experiment configuration via [`ExperimentOptions`](src/controller/options.py)

### 4. Data Generation
- Synthetic ground-truth generation
- Document processing capabilities
- Support for multiple input formats

## Installation

1. Clone the repository
2. Install the package:
```bash
cd eval
pip install .
```

## Configuration

The framework uses environment variables for Couchbase and OpenAI configurations. Refer [`.env-template`](.env-template).

## Usage

Example usage provided in [`example.ipynb`](examples/rag_eval.ipynb)

### Synthetic Data Generation

The framework provides tools to generate synthetic question-answer pairs from your documents, which can be used as ground truth for evaluation. <span style="color:yellow">For json and csv documents, provide detailed metadata including the dataset schema for accurate data generation.</span>

For perf logging (memory and runtime):
~~~sh
~$ python3 -m src.data.generator --path <path to the csv/json file> --metadata-file <path to metadata txt file> --field <field name in json to use (optional)> --limit <limit number of rows to process (optional)> --format <file format ('csv' or 'json')>
~~~

#### From CSV Files

```python
from src.data.generator import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator()

# Generate synthetic data from a CSV file
metadata = "Document contains product descriptions with fields: name, description, price, and category."
generated_data = generator.synthesize_from_csv(
    path="data/products.csv",
    field="description",  # Optional: specify which field to use
    metadata=metadata
)

# Access the generated data
questions = generated_data["questions"]
answers = generated_data["answers"]
reference_contexts = generated_data["reference_contexts"]

# Print sample data
for question, answer, context in zip(questions, answers, reference_contexts):
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print("---")
```

#### From JSON Documents

```python
from src.data.generator import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator()

# Load documents from JSON
documents = generator.load_from_json(
    path="data/documents/",  # Can be a directory of JSON files or a single JSON file
    field="content"  # Optional: specify which field to use
)

# Generate synthetic data
metadata = "Documents are technical articles about machine learning."
generated_data = generator.synthesize(
    documents=documents,
    metadata=metadata
)

# Use the generated data to create an evaluation dataset
from src.data.dataset import EvalDataset

eval_dataset = EvalDataset(
    questions=generated_data["questions"],
    answers=generated_data["answers"],
    reference_contexts=generated_data["reference_contexts"],
    # You'll need to add responses and retrieved_contexts after running your RAG system
)

# Save the dataset
eval_dataset.to_json("datasets/synthetic_eval_dataset.json")
```

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
from src.evaluator.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size

engine = ValidationEngine(
    dataset=dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size] # Calculates a set of default metrics if metrics are not provided
)
results = engine.evaluate()
```
Results are stored in folder named `.results`

### Experiment based evaluation

```python
from src.controller.options import ExperimentOptions
from src.controller.manager import Experiment

# Configure experiment
experiment_options = ExperimentOptions(
    experiment_id="exp_001",
    dataset_id="dataset_001",
    metrics=[context_precision, context_recall, faithfulness],
    chunk_size=100,
    chunk_overlap=20,
    embedding_model="text-embedding-3-large",
    embedding_dimension=3072,
    llm_model="gpt-4"
)

# Create experiment, results are stored
experiment = Experiment(dataset=dataset, options=experiment_options) #Pulls the dataset from the couchbase cluster using `dataset_id` provided in `experiment_options` if dataset not provided.

# Load the experiment config and results to couchbase kv cluster
experiment.load_to_couchbase()

# Retrieve experiment results from the couchbase kv store
experiment_result = Experiment().retrieve(experiment_id="exp_001")
```
Results are stored in `.results-<experiment_id>`


## Roadmap

1. **Basic features**
   - Custom metric integration
   - Multi-turn conversation evaluation
   - Composite, multi-hop question answer generator.
   - Generating multiple ground truth answers for a given document

2. **Multimodal RAG Support**
   - Image retrieval evaluation
   - Table content processing
   - Cross-modal metrics

3. **Agentic Evaluation**
    - Tool call evaluation.
    - Node transition evaluation.