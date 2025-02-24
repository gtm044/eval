# eval
Framework to evaluate RAG systems and synthesize ground truth data.

## Features

- **EvalDataset Data Class**: 
  - Define and manage evaluation datasets with fields for questions, answers, responses, reference contexts, and retrieved contexts.
  - Validate that all input lists are of the same length.
  - Converts datasets to JSON format.

- **LoadOperator**:
  - Loads processed documents into a Couchbase keyspace.
  - Retrieves documents from Couchbase and returns them as an `EvalDataset` object.
  - Automatically handles connection and authentication to Couchbase using environment variables.

- **SyntheticDataGenerator**:
  - Generates synthetic questions and answers for provided documents using language models.
  - Expands documents to create larger datasets.
  - Synthesizing ground truth data from raw documents.


## Roadmap
- `EvalDataset` data class - Done
    - Input:  Questions, Ground truth answers, Reference Contexts, Retrieved Contexts, Generated Responses (or subset of these fields)
    - Ingest from python lists/dictionaries.

- Operator to load the ground truth data into a key value store. - Done
    - Input: `EvalDataset` data class.

- Operator to retrieve the ground truth data from a key value store. - Done

- `Synthesize`: Synthesize gt from raw documents
    - Provide document directly as json or ingest from cloud.
    - Chunk the documents
    - Generate question answer pair for each chunked document
    - If possible, generate a ground truth for the chunks -> induces too much uncertainty.
    - Refer ARES (https://github.com/stanford-futuredata/ARES). Might not be able to integrate ARES directly as it is, might have to tweak the implementation a little bit, not sure if thats possible.
    - Method `expand` to expand the given raw documents if the data is small. (Currently using paraphrasing, byut very vague. Try to find other methods to do the same.)
    - ARES - single pipeline from top to bottom. Almost the same thing that we are trying to do. And no dataset expansion implementation.
    - Ideas on expansion:
        - Text AutoAugment (TAA) - https://github.com/lancopku/text-autoaugment
        - Ading noise to text embedddings, conerting the noise back to text using vec2text models. -> Need to quantify the noisy text, might not be similar to the initial document.

- `ValidationOptions`: Metric configurations
    - Choose specific metrics/overall system evaluation/segment evaluation (chunking, retrieveal, generation etc)
    - Validate if the metric provided is in the list of implemented metrics.
    - ...

- For multimodel RAGs:
    - What if the document is an image or a table? 
    - What could the user provide as the input? - Instead of reference contexts, the images and the retrieved images instead of the retrieved contexts.
    - How to store images -> numpy array. Evaluation can be performed by comparing the retrieved and reference images using basic array matching. 
    - Very simple metric for image retrieveal evaluation - accuracy (binary classification).
    - Tables: Compilcated, each chunk can be a single row or a set of rows. For retrieveal evaluation, can just match the rows as it is, again binary classicfication metrics. For responses, compare the ground truth and the reference answers, simiarity measures bw the generated response and the retrived document wont work for tables.
    - For synthesizing ground truths for tables, simple prompt engineering is enough, with the serialized json of the table/required row given as the input.

- Ingesting raw data to synthesize ground truth:
    - Structured Formats: CSV, JSON (How to infer schema?), Parquet.
    - Unstructured Formats: PDFs
    - Logs: ?
    - File metadata can be provided as the document description.