# src/utils/prompts.py
synthetic_query_prompt: str = (
    "You are an expert question-answering system provided with a document. You must create a question for the provided document. "
    "The question must be answerable within the context of the document and should directly ask about information contained in the document.\n\n"
    "The document might also be a json object converted to a string, with each row of the dataset being a json object. If it is a json object, a metadata will be provided along with the document. "
    "In this case, you should create a question that can be answered using the information in the json object.\n\n"
    "Important: Do NOT start with phrases like 'What does this document say about...' or 'According to the document...'. "
    "Instead, ask direct questions such as 'When was Paris founded?' or 'How does the fox move in the story?'\n\n"
    "The response should contain only the question without any additional text or formatting.\n\n"
)

# Prompt to generate well-formed answers.
synthetic_valid_answer_prompt: str = (
    "You are an expert question-answering system provided with a <question> and a <document>. You must generate an answer for the provided question. "
    "The answer must be answerable within the context of the document. "
    "The document might be a JSON object converted to a string, with each row of the dataset being a JSON object. If it is a JSON object, use the information in the JSON to answer the question. A metadata will be provided along with the document if the document is a JSON object representation.\n\n"
    "Generate a complete, well-formed sentence that directly answers the question. The answer should be concise but should not be a single word or phrase. Instead, it should be structured as a meaningful sentence incorporating necessary details from the document.\n\n"
)

# Prompt to generate short and specific answers.
# synthetic_valid_answer_prompt: str = (
#     "You are an expert question-answering system provided with a <question> and a <document>. You must generate an answer for the provided question. "
#     "The answer must be answerable within the context of the document. "
#     "The document might be a json object converted to a string, with each row of the dataset being a json object. If it is a json object, use the information in the json to answer the question. A metadata will be provided along with the document if the document is a json object representation.\n\n"
#     "Generate the answer alone without any additional information.\n\n"
# )

expand_documents_prompt: str = (
    "You are an expert in document expansion. You are provided with a document that needs to be expanded. \n\n"
    "This is done to create a bigger dataset given a small document to test a RAG system.\n\n"
    "The document might be a json object converted to a string, representing a row of a dataset. If it is a json object, expand the textual fields while preserving the structure of the json.\n\n"
    "Expanding a document involves generating multiple variations of a given document with different wording, sentence structure, and phrasing while ensuring the core content remains intact. The output should maintain coherence, factual accuracy, and natural readability. \n\n"
    "The response should contain only the expanded document according to the json schema given below without any additional text, formatting or whitespaces.\n\n"
    "The output should be a json string containing a dictionary with the list of expanded documents, dont include the input document in the response.\n\n"
    "The number of documents to generate is not fixed and is decided by the content of the document, if you reach a point where you can't paraphrase anymore to create more meaningful documents, stop.\n\n"
    "But if there is a number specified in an input field <limit>, you should generate that many documents.\n\n"
)