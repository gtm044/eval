# eval
Framework to evaluate RAG systems and synthesize ground truth data.

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
    - Method `expand` to expand the given raw documents if the data is small.

- Ingesting raw data to synthesize ground truth:
    - Structured Formats: CSV, JSON (How to infer schema?), Parquet.
    - Unstructured Formats: PDFs
    - Logs: ?
    - File metadata can be provided as the document description.