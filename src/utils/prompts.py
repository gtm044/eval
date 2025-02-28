synthetic_query_prompt: str = (
    "You are an expert question-answering system provided with a document. You must create a question for the provided document. "
    "The question must be answerable within the context of the document.\n\n"
    "The response should contain only the question without any additional text or formatting.\n\n"
)
synthetic_valid_answer_prompt: str = (
    "You are an expert question-answering system provided with a <question and a <document>. You must create an answer for the provided question. "
    "The answer must be answerable within the context of the document. "
    "Provide the answer alone without any additional text or formatting.\n\n"
)
expand_documents_prompt: str = (
    "You are an expert in document expansion. You are provided with a document that needs to be expanded. \n\n"
    "This is done to create a bigger dataset given a small document to test a RAG system.\n\n"
    "Expanding a document involves generating multiple variations of a given document with different wording, sentence structure, and phrasing while ensuring the core content remains intact. The output should maintain coherence, factual accuracy, and natural readability. \n\n"
    "The response should contain only the expanded document according to the json schema given below without any additional text, formatting or whitespaces.\n\n"
    "The output should be a json string containing a dictionary with the list of expanded documents, dont include the input document in the response.\n\n"
    "The number of documents to generate is not fixed and is decided by the content of the document, if you reach a point where you can't paraphrase anymore to create more meaningful documents, stop.\n\n"
    "But if there is a number specified in an input field <limit>, you should generate that many documents.\n\n"
)