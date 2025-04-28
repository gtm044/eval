from typing import List, Any
from openai import OpenAI
import os
import json
from tqdm import tqdm
import platform
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def doc_to_nl(documents: List[Any]) -> str:
    EXTRACTION_PROMPT = """
    Convert the following JSON to natural language english sentences.
    These natural language sentences are used to extract entities and relationships from the document.

    JSON Dictionary
    {json}

    NATURAL LANGUAGE
    """.strip()
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    outputs = []
    
    # Store chat history for context
    chat_history = []
    max_history = 10  # Keep last 10 messages for context
    
    for document in tqdm(documents, desc="Processing entities: "):
        prompt = EXTRACTION_PROMPT.format(json=document)
        
        # Create messages with history
        chat_messages = chat_history + [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = chat_messages,
            temperature = 0.0,
        )
        
        output = response.choices[0].message.content.strip()
        outputs.append(output)
        
        # Update chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": output})
        
        # Keep only the last max_history messages
        if len(chat_history) > max_history:
            chat_history = chat_history[-max_history:]
            
    return outputs

def text_to_relations_llm(texts: List[str]) -> List[List[Any]]:
    RELATION_EXTRACTION_PROMPT = """
    Convert the following text into a list of meaningful (head, relation, tail) triples.

    Use specific and informative relation types instead of generic terms like "is" or "has".
    Choose relation verbs that describe the nature of the relationship, such as:
    - "has_name", "has_type", "located_in", "includes", "owns", "teaches", "lives_in", "has_feature", etc.

    Follow this output format:
    [
    {{"head": "entity1", "type": "relation", "tail": "entity2"}},
    {{"head": "entity3", "type": "relation", "tail": "entity4"}}
    ]

    Do not include any explanation or extra text. Only return the JSON array.

    Input Text:
    {text}

    OUTPUT
    """.strip()

    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    relations = []
    
    # Store chat history for context
    chat_history = []
    max_history = 5  # Keep last 10 messages for context
    prompt = RELATION_EXTRACTION_PROMPT
    for text in tqdm(texts, desc="Extracting relations: "):
        def get_response(text):
            prompt = RELATION_EXTRACTION_PROMPT.format(text=text)
            
            # Create messages with history
            chat_messages = chat_history + [{"role": "user", "content": prompt}]
            
            response = client.chat.completions.create(
                model = "gpt-4o",
                messages = chat_messages,
                temperature = 0.0,
            )
            
            return response.choices[0].message.content.strip()
        
        max_attempts = 10
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            try:
                output = get_response(text)
                output = output.replace("```json", "").replace("```", "")
                relations.append(json.loads(output))
                success = True
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    output = get_response(text)
                    return output
        # Update chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": output})
        
        # Keep only the last max_history messages
        if len(chat_history) > max_history:
            chat_history = chat_history[-max_history:]
            
    return relations
    

def text_to_relations(texts: List[str]) -> List[Any]:
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    gen_kwargs = {
        "max_length": 8192,
        "length_penalty": 0,
        "num_beams": 15,
        "num_return_sequences": 15,
    }
    raw_relations = []
    for text in tqdm(texts, desc="Extracting relations: "):
        model_inputs = tokenizer([text], max_length=512, return_tensors="pt", padding=True, truncation=True)
        generated_tokens = model.generate(**model_inputs, **gen_kwargs)
        decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        raw_relations.append(decoded_output)
    return process_relations(raw_relations)


def process_relations(total_relations: List[List[Any]]):
    processed_relations = []
    for document_relations in total_relations:
        processed_relation = [] 
        for relation in document_relations:
            extracted_relations = parse_relations(relation)
            processed_relation.extend(extracted_relations)
        # Remove the duplicates in the processed relation
        unique_relations = {}
        for rel in processed_relation:
            rel_tuple = tuple(sorted(rel.items()))
            unique_relations[rel_tuple] = rel
        processed_relation = list(unique_relations.values())
        processed_relations.append(processed_relation)
    return processed_relations


def parse_relations(text: str) -> List[dict]:
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

def get_default_save_directory():
    """
    Get the default directory for saving trace files based on the operating system.
    
    Returns:
        Path: The platform-specific default directory
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        default_dir = os.path.expanduser("~/Documents/eval_clusters")
    elif system == "Windows":
        default_dir = os.path.join(os.path.expanduser("~"), "Documents", "eval_clusters")
    else:  # Other unix
        default_dir = os.path.expanduser("~/eval_clusters")
    # Create directory if it doesn't exist
    os.makedirs(default_dir, exist_ok=True)
    return Path(default_dir)