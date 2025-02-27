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

## Notes

- Add a bash to download the spacy corpus on installing the sdk. ``python -m spacy download en_core_web_sm``
- Implementing bleu score from scratch (as of now imported from the nltk library)
- About the jaccard index for chunking evaluation, ground truth chunks and whether we shouold include ground truth chunk synthesizing in the synthesize class.
- For list metric output, handle the output dictionary from the ValidationEngine.
- Research on report generation. -> Either we use an LLM (give it the otutput dictionary and prompt engineer) or we can design a schema to include descriptions, a score range and the issue description.


## Roadmap

- Method `expand` to expand the given raw documents if the data is small. (Currently using paraphrasing, byut very vague. Try to find other methods to do the same.)
- ARES - single pipeline from top to bottom. Almost the same thing that we are trying to do. And no dataset expansion implementation.
- Ideas on expansion:
    - Text AutoAugment (TAA) - https://github.com/lancopku/text-autoaugment
    - Ading noise to text embedddings, conerting the noise back to text using vec2text models. -> Need to quantify the noisy text, might not be similar to the initial document.
- Report Generation

- Experiment management
  - Store the evaluation results to a kv store
  - Each result should have a metadata with the experiment options.
  - on calling the add function, an output directory is created with the experiment_<experiment_id> as the name and store al the output files inside (output.json, output.csv, averaged_output.csv, averaged_output.json, experiment_config.json)
  - Experiment Config is the metadata, contains the Experiment Options and the timestamp, the length of the ground truth dataset and the metrics used and the dataset description.
  - Load the contents of the directory to the couchbase kv store with the metadata with each dictionary content in the ValidationEngine output.

TO-DO:

- For multimodel RAGs (Future work):
    - What if the document is an image or a table? 
    - What could the user provide as the input? - Instead of reference contexts, the images and the retrieved images instead of the retrieved contexts.
    - How to store images -> numpy array. Evaluation can be performed by comparing the retrieved and reference images using basic array matching. 
    - Very simple metric for image retrieveal evaluation - accuracy (binary classification).
    - Tables: Compilcated, each chunk can be a single row or a set of rows. For retrieveal evaluation, can just match the rows as it is, again binary classicfication metrics. For responses, compare the ground truth and the reference answers, simiarity measures bw the generated response and the retrived document wont work for tables.
    - For synthesizing ground truths for tables, simple prompt engineering is enough, with the serialized json of the table/required row given as the input.
    - Method for the users to define metrics.
    - Remove the averaged metrics.
    - Work on the experiment schema.


- Ingesting raw data to synthesize ground truth:
    - Structured Formats: CSV, JSON (How to infer schema?), Parquet.
    - Unstructured Formats: PDFs
    - Logs: ?
    - File metadata can be provided as the document description.