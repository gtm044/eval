# src/utils/prompts.py

import os
import jinja2

# Set up Jinja environment
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils/prompt_templates')
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(template_dir),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)

# Load templates
synthetic_query_template = jinja_env.get_template('synthetic_query_prompt.jinja')
synthetic_valid_answer_template = jinja_env.get_template('synthetic_valid_answer_prompt.jinja')
synthetic_valid_short_answer_template = jinja_env.get_template('synthetic_valid_short_answer_prompt.jinja')
expand_documents_template = jinja_env.get_template('expand_documents_prompt.jinja')

# Get prompt strings (default rendering without parameters)
synthetic_query_prompt = synthetic_query_template.render()
synthetic_valid_answer_prompt = synthetic_valid_answer_template.render()
synthetic_valid_short_answer_prompt = synthetic_valid_short_answer_template.render()
expand_documents_prompt = expand_documents_template.render()

# Functions to render templates with parameters
def render_synthetic_query_prompt(**kwargs):
    """Render the synthetic query prompt with parameters."""
    return synthetic_query_template.render(**kwargs)

def render_synthetic_valid_answer_prompt(**kwargs):
    """Render the synthetic valid answer prompt with parameters."""
    return synthetic_valid_answer_template.render(**kwargs)

def render_synthetic_valid_short_answer_prompt(**kwargs):
    """Render the synthetic valid short answer prompt with parameters."""
    return synthetic_valid_short_answer_template.render(**kwargs)

def render_expand_documents_prompt(**kwargs):
    """Render the expand documents prompt with parameters."""
    return expand_documents_template.render(**kwargs)

# Original hardcoded prompts for reference
"""
synthetic_query_prompt: str = (
    "You are an expert question-answering system provided with a document. You must create a natural, direct question based on information in the document.\n\n"
    
    "IMPORTANT: Generate questions based EXACTLY on what the document says, even if the information is factually incorrect. For example, if the document states 'The moon is made of cheese', your question should be 'What is the moon made of?' and NOT 'What is the composition of the moon according to scientific evidence?'\n\n"
    
    "❌ ABSOLUTELY NEVER USE PHRASES LIKE:\n"
    "- 'according to the document'\n"
    "- 'what does the document say about'\n"
    "- 'as mentioned in the document'\n"
    "- 'based on the document'\n"
    "- 'in the document'\n"
    "- ANY phrase that references 'the document', 'the text', 'the passage', or similar terms\n\n"
    
    "❌ NEVER question, correct, or avoid false information in the document. Treat ALL information in the document as valid material for questions.\n\n"
    
    "✅ INSTEAD, ASK DIRECT QUESTIONS LIKE:\n"
    "- 'Where is the Great Wall of China located?'\n"
    "- 'What is the capital of France?'\n"
    "- 'When was the Declaration of Independence signed?'\n"
    "- 'How many planets are in our solar system?'\n\n"
    
    "Your questions should sound natural, as if asked by a person who wants to know the information, not as if they're testing knowledge of a specific document.\n\n"
    
    "If the document does not contain enough information to generate a question, respond with just 'NO_QUESTION_POSSIBLE'\n\n"
    
    "The response should contain only the question without any additional text or formatting, unless you determine no question is possible.\n\n"
    
    "Note: The document provided might be a JSON object converted to a string. If so, create a question that can be answered using the information in the JSON object, following the same rules above."
)

# Prompt to generate well-formed answers.
synthetic_valid_answer_prompt: str = (
    "You are an expert question-answering system provided with a <question> and a <document>. You must generate an answer for the provided question. "
    "The answer must be answerable within the context of the document. "
    "The document might be a JSON object converted to a string, with each row of the dataset being a JSON object. If it is a JSON object, use the information in the JSON to answer the question. A metadata will be provided along with the document if the document is a JSON object representation.\n\n"
    "Generate a complete, well-formed sentence that directly answers the question. The answer should be concise but should not be a single word or phrase. Instead, it should be structured as a meaningful sentence incorporating necessary details from the document.\n\n"
    "If the question is marked as 'NO_QUESTION_POSSIBLE' or if the document does not contain enough information to answer the question provoded, respond with just 'NO_ANSWER_POSSIBLE'.\n\n"
)

# Prompt to generate short and specific answers.
# synthetic_valid_short_answer_prompt: str = (
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
"""

# Example 

# print(render_synthetic_query_prompt(
#     custom_instructions="",
#     example_questions=["What is the capital of France?", "What is the capital of Germany?"],
#     no_question_response="NO_QUESTION_POSSIBLE"
# ))

# print(synthetic_valid_answer_prompt)
