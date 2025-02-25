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
- Add the averaged metrics (avg_chunk_size and retrieval accuracy as of now) as a seperate dictionary object on appending to the final output list in the ValidationEngine. 
- Probably change the name of retrieval accuracy to combined retrieveal accuracy and add these two metrics and the average of all other metrics as a seperate dictionary, along with the weighted index score for each segment.


## Roadmap
- `EvalDataset` data class - Done
    - Input:  Questions, Ground truth answers, Reference Contexts, Retrieved Contexts, Generated Responses (or subset of these fields)
    - Ingest from python lists/dictionaries.

- Operator to load the ground truth data into a key value store. - Done
    - Input: `EvalDataset` data class.

- Operator to retrieve the ground truth data from a key value store. - Done

- `Synthesize`: Synthesize gt from raw documents - Done
    - Provide document directly as json.
    - Chunk the documents
    - Generate question answer pair for each chunked document
    - If possible, generate a ground truth for the chunks -> induces too much uncertainty.
    - Refer ARES (https://github.com/stanford-futuredata/ARES). Might not be able to integrate ARES directly as it is, might have to tweak the implementation a little bit, not sure if thats possible.
    - Method `expand` to expand the given raw documents if the data is small. (Currently using paraphrasing, byut very vague. Try to find other methods to do the same.)
    - ARES - single pipeline from top to bottom. Almost the same thing that we are trying to do. And no dataset expansion implementation.
    - Ideas on expansion:
        - Text AutoAugment (TAA) - https://github.com/lancopku/text-autoaugment
        - Ading noise to text embedddings, conerting the noise back to text using vec2text models. -> Need to quantify the noisy text, might not be similar to the initial document.

- Loading the raw documents from json documents - Done.
- Provide a folder containing the json documents, and the field to consider - Done.
- If it is a single json document with a list of json objects, provide an extra argument which allows the system to understand which schema it follows - Done.

- `ValidationOptions`: Metric configurations
    - Choose specific metrics/overall system evaluation/segment evaluation (chunking, retrieveal, generation etc)
    - Validate if the metric provided is in the list of implemented metrics.
    - ...
    - How to specify whether to get the mentioned metrics or perform the entire evaluation?

- For multimodel RAGs (Future work):
    - What if the document is an image or a table? 
    - What could the user provide as the input? - Instead of reference contexts, the images and the retrieved images instead of the retrieved contexts.
    - How to store images -> numpy array. Evaluation can be performed by comparing the retrieved and reference images using basic array matching. 
    - Very simple metric for image retrieveal evaluation - accuracy (binary classification).
    - Tables: Compilcated, each chunk can be a single row or a set of rows. For retrieveal evaluation, can just match the rows as it is, again binary classicfication metrics. For responses, compare the ground truth and the reference answers, simiarity measures bw the generated response and the retrived document wont work for tables.
    - For synthesizing ground truths for tables, simple prompt engineering is enough, with the serialized json of the table/required row given as the input.

- Chunking metrics - Done:
    - How to interpret each chunk? Tokenize? Splitting into words doesn't make sense. If we tokenize, will the reference chunks and the generated chunks ever contain the same tokens?
    - Average chunk size: Tokenize each chunk, find the average of all the chunks. Lets say there are 10 documents (10 chunks), find the length of the tokens of each document, get the average.
    - For IoU(Jaccard Index), we need the relevant and the retrieved chunks, how would a user get the relevant chunks?  
    - For the avg_chunk_size, we need a threshold to determine the chunk size that is admissible. User defined or programmed?
    - Use a normalized parabolic function to get an index value for the average chunk size.

- Retrieval metrics:
    - For contet scores, ROUGE scores can be either written from scratch, or loaded from the hugggingface evaluate library.


- Ingesting raw data to synthesize ground truth:
    - Structured Formats: CSV, JSON (How to infer schema?), Parquet.
    - Unstructured Formats: PDFs
    - Logs: ?
    - File metadata can be provided as the document description.