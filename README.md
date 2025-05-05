# Agent Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) and Agentic systems using RAGAS and synthesizing ground truth data from raw documents.

## Overview

This framework provides tools and metrics to evaluate RAG/AI systems across three key components:
- Retrieval evaluation
- Generation evaluation
- Agentic workflow evaluation

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

A complete workflow example for a RAG use case provided in [`example.ipynb`](examples/rag_eval.ipynb)

### Synthetic Data Generation

The framework provides tools to generate synthetic question-answer pairs from your documents, which can be used as ground truth for evaluation. <span style="color:yellow">For json and csv documents, provide detailed metadata including the dataset schema for accurate data generation.</span>

For generation and performance stats for the single-hop generator (memory and runtime):
~~~sh
~$ python3 -m src.data.generator --path <path to the csv/json file> --metadata-file <path to metadata txt file> --field <field name in json to use (optional)> --limit <limit number of rows to process (optional)> --format <file format ('csv' or 'json')>
~~~

**Note**: Set the OPENAI_API_KEY in the environment variables (.env) file.

#### From CSV Files

```python
from eval.src.data.generator import init_generator

# Initialize the generator
generator = init_generator(multi_hop=False) # multi-hop=True for complex multi-hop data generator

# Generate synthetic data from a CSV file
metadata = "Document contains product descriptions with fields: name, description, price, and category."
generated_data = generator.synthesize_from_csv(
    path="data/products.csv",
    field="description",  # Optional: specify which field to use
    limit=10, # Optional: limit the number of rows to process
    metadata=metadata,
    output_path = "generation.json" # Optional, will use a default path if not provided
)

# Access the generated data
output_json = json.load(open("generation.json"))
questions = [d["question"] for d in output_json]
answers = [d["answers"] for d in output_json]
reference_contexts = [d["reference"] for d in output_json]
# or
questions, answers, reference_contexts = generated_data["questions"], generated_data["answers"], generated_data["reference_contexts"]

# Print sample data
for question, answer, context in zip(questions, answers, reference_contexts):
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print("---")
```

#### From JSON Documents

```python
from eval.src.data.generator import init_generator

# Initialize the generator
generator = init_generator(multi_hop=False) # multi-hop=True for complex multi-hop data generator       

# Generate synthetic data directly from JSON files
metadata = "Documents are technical articles about machine learning."
generated_data = generator.synthesize_from_json(
    path="data/documents/",  # Can be a directory of JSON files or a single JSON file
    field="content",  # Optional: specify which field to use
    limit=10, # Optional: limit the number of rows to process
    metadata=metadata,
    output_path = "generation.json" # Optional, will use a default path if not provided
)

# Access the generated data
output_json = json.load(open("generation.json"))
questions = [d["question"] for d in output_json]
answers = [d["answers"] for d in output_json]
reference_contexts = [d["reference"] for d in output_json]
# or
questions, answers, reference_contexts = generated_data["questions"], generated_data["answers"], generated_data["reference_contexts"]

# Print sample data
for question, answer, context in zip(questions, answers, reference_contexts):
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Context: {context}")
    print("---")

```
---
### Basic Evaluation for a RAG system or simple AI pipeline

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

# Run evaluation with metrics (can also provide RAGAS metrics)
from eval.src.evaluator.metrics import context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size

engine = ValidationEngine(
    dataset=dataset,
    metrics=[context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness, avg_chunk_size] # Calculates a set of default metrics if metrics are not provided
)
results = engine.evaluate()
```
---
### Basic Evaluation for an Agentic System

```python
from eval.src.data.dataset import EvalDataset
from eval.src.evaluator.validation import ValidationEngine

# Create dataset
dataset = EvalDataset(
    questions=["What is the price of copper?", "What is the price of gold?"],
    agent_responses=[["The current price of copper is $0.0098 per gram."], ["The current price of gold is $88.16 per gram."]],
    agent_tool_calls=[
        [{"name": "get_price", "args": {"item": "copper"}}],
        [{"name": "get_price", "args": {"item": "gold"}}]
    ],
    agent_tool_outputs=[["$0.0098"], ["$88.16"]],
    reference_tool_calls=[
        [{"name": "get_price", "args": {"item": "copper"}}],    
        [{"name": "get_price", "args": {"item": "gold"}}]
    ],
    gt_answers=[["$0.0098 per gram"], ["$88.16 per gram"]],
    gt_tool_outputs=[["$0.0098"], ["$88.16"]]
)

# Run evaluation with metrics
from eval.src.evaluator.metrics import tool_call_accuracy, tool_accuracy, agent_response_correctness

engine = ValidationEngine(
    dataset=dataset,
    metrics=[tool_call_accuracy, tool_accuracy, agent_response_correctness]
)
results, metrics_used, schema, avg_metrics = engine.evaluate() 
```

Results are stored in `.results` folder in the current working directory.

---
### Experiment based evaluation

```python
from eval.src.controller.options import ExperimentOptions
from eval.src.controller.manager import Experiment

# Configure experiment
experiment_options = ExperimentOptions(
    experiment_id="exp_001",
    dataset_id="dataset_001",
    metrics=[tool_call_accuracy, tool_accuracy, agent_response_correctness],
    embedding_model="text-embedding-3-large",
    embedding_dimension=3072,
    llm_model="gpt-4o",
    experiment_description="Experiment for agentic evaluation",
    # Can add more fields as required...
)

# Create experiment, results are stored
experiment = Experiment(dataset=dataset, options=experiment_options) #Pulls the dataset from the couchbase cluster using `dataset_id` provided in `experiment_options` if dataset not provided.

# Load the experiment config and results to couchbase kv cluster
experiment.load_to_couchbase()

# Retrieve experiment results from the couchbase kv store
experiment_result = Experiment().retrieve(experiment_id="exp_001", collection="results")
```
Results are stored in `.results-<experiment_id>`

---
### Adding a custom user defined metric
Custom metrics can be added as functions to the list of metrics provided to the `ExperimentOptions` object. The functions should follow the below signature:
```txt
Arguments should belong to the following list of attributes of the `EvalDataset` object. Each argument is a list of values over the entire dataset and should be present in the `EvalDataset` object: 
    - questions: List[str]
    - responses: List[str]
    - answers: List[List[str]]
    - reference_contexts: List[str]
    - retrieved_contexts: List[List[str]]
    - agent_responses: List[List[str]]
    - agent_tool_calls: List[List[Dict[str, Any]]]
    - agent_tool_outputs: List[List[str]]
    - reference_tool_calls: List[List[Dict[str, Any]]]
    - gt_answers: List[List[str]]
    - gt_tool_outputs: List[List[str]]
Output:
    - List of metric values (one per data instance evaluated)
```
Refer [validation.py](src/evaluator/validation.py) for an example of a custom metric.
Example dummy metric:
```python
# Define the custom metric
def custom_metric(questions: List[str], responses: List[str]):
    metric_values = []
    for question, response in zip(questions, responses):
        metric_values.append(1 if question == response else 0)
    return metric_values
```

---
### Agent Evaluation with LangGraph

Example provided in [`langgraph_eval.py`](examples/agent_langgraph_improved.py)

---
### Tracing LangGraph and LangChain

```python
# LangGraph tracing
from eval.src.langgraph.trace_v2 import create_traced_agent, log_traces

# <agent implementation>
react_graph = builder.compile()

# Create a traced version of the agent
traced_graph = create_traced_agent(react_graph)

def get_agent_response_wrapped(queries):
    """Use the TracedAgent wrapper to automatically trace all interactions"""
    results = []
    for query in queries:
        messages = [HumanMessage(content=query)]
        result = traced_graph.stream({"messages": messages})
        results.append(list(result))  
    
    log_path = log_traces()
    print(f"Traces saved to: {log_path}")
    
    return results

# Example usage
get_agent_response_wrapped(["What is the price of copper?", "What is the price of gold?"])

# LangChain tracing (for RAG systems)
from eval.src.langchain.trace import interceptor

agent = Agent(..., callbacks=[interceptor])
agent.invoke({"input": "What is the price of gold?"})
interceptor.log()
```

## Roadmap

   - Multi-turn conversation evaluation
   - Image/structured data retrieval evaluation
   - Cross-modal metrics
   - Node transition evaluation