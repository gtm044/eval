# RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using RAGAS and synthesizing ground truth data from raw documents.

## Overview

This framework provides tools and metrics to evaluate RAG systems across three key components:
- Chunking evaluation
- Retrieval evaluation
- Generation evaluation
- Agentic evaluation

The framework integrates with RAGAS, a popular RAG evaluation library, and provides a structured approach for experiment management, storage and result persistence. It also supports tracing and evaluating agentic pipelines built with LangChain and LangGraph.

<!-- ## Features

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

#### Agentic Metrics
- Tool call accuracy - Compares agent's tool calls with reference tool calls
- Answer correctness - Compares final AI responses with ground truth answers
- Answer faithfulness - Checks if AI's answer is faithful to the tool outputs
- Tool accuracy - Compares tool outputs with ground truth tool outputs

### 3. Experiment Management
- Structured experiment tracking with unique experiment IDs
- Metadata management including chunking, embedding, and LLM parameters
- Result persistence with Couchbase integration
- Flexible experiment configuration via [`ExperimentOptions`](src/controller/options.py)

### 4. Data Generation
- Synthetic ground-truth generation
- Document processing capabilities
- Support for multiple input formats

### 5. Agent Tracing and Evaluation
- Support for LangGraph trace logging and analysis
- LangChain tracing with comprehensive event capture
- Agent validation engine for evaluating LLM agents -->

## Installation

1. Clone the repository
2. Install the package:
```bash
cd eval
pip install .
```

## Configuration

The framework uses environment variables for Couchbase and OpenAI configurations. Refer [`.env.template`](.env.template).

## Usage

A complete workflow example provided in [`example.ipynb`](examples/rag_eval.ipynb)

### Synthetic Data Generation

The framework provides tools to generate synthetic question-answer pairs from your documents, which can be used as ground truth for evaluation. <span style="color:yellow">For json and csv documents, provide detailed metadata including the dataset schema for accurate data generation.</span>

For generation and perf logging (memory and runtime):
~~~sh
~$ python3 -m src.data.generator --path <path to the csv/json file> --metadata-file <path to metadata txt file> --field <field name in json to use (optional)> --limit <limit number of rows to process (optional)> --format <file format ('csv' or 'json')>
~~~

**Note**: Set the OPENAI_API_KEY in the environment variables (.env) file.

#### From CSV Files

```python
from eval.src.data.generator import SyntheticDataGenerator

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
from eval.src.data.generator import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator()

# Generate synthetic data directly from JSON files
metadata = "Documents are technical articles about machine learning."
generated_data = generator.synthesize_from_json(
    path="data/documents/",  # Can be a directory of JSON files or a single JSON file
    field="content",  # Optional: specify which field to use
    metadata=metadata
)

# Access the generated data
questions = generated_data["questions"]
answers = generated_data["answers"]
reference_contexts = generated_data["reference_contexts"]

# Use the generated data to create an evaluation dataset
from eval.src.data.dataset import EvalDataset

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
from eval.src.data.dataset import EvalDataset
from eval.src.evaluator.validation import ValidationEngine

# Create dataset
dataset = EvalDataset(
    questions=["What is RAG?"],
    answers=[["RAG is a retrieval-augmented generation system"]],
    responses=["RAG combines retrieval with generation"],
    reference_contexts=["RAG systems use retrieval to enhance generation"],
    retrieved_contexts=[["RAG: retrieval-augmented generation"]]
)

# Run evaluation with RAGAS metrics
from eval.src.evaluator.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size

engine = ValidationEngine(
    dataset=dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size] # Calculates a set of default metrics if metrics are not provided
)
results = engine.evaluate()
```
Results are stored in folder named `.results`

### Experiment based evaluation

```python
from eval.src.controller.options import ExperimentOptions
from eval.src.controller.manager import Experiment

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

### Agent Evaluation

Example provided in [`agentic_evaluation`](examples/agent_langgraph.py)

### Tracing LangGraph and LangChain

```python
# LangGraph tracing
from eval.src.langgraph.trace import track_variable, log_lang_stream

@track_variable("stream") # Provide the variable that stores the result of the .stream function call
def run_agent(query):
    # Your LangGraph agent implementation
    # ...
    return stream

# Run the agent
run_agent("What is the price of copper?")

# Save the logs
log_lang_stream()

# LangChain tracing
from eval.src.langchain.trace import interceptor

# Create a LangChain agent with the interceptor
agent = Agent(..., callbacks=[interceptor])

# Run the agent
agent.invoke({"input": "What is the price of gold?"})

# Save the logs
interceptor.log()
```

## Roadmap

1. **Advanced features**
   - Multi-turn conversation evaluation
   - Composite, multi-hop question answer generator

2. **Multimodal RAG Support**
   - Image retrieval evaluation
   - Table content processing
   - Cross-modal metrics

3. **Agentic Evaluation**
   - Advanced tool call evaluation
   - Node transition evaluation
   - Context-aware evaluation for complex agentic workflows