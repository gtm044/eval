# LLM to parse langchain logs

from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from src.utils.prompts import parse_langchain_logs_prompt, render_parse_langchain_logs_prompt
from src.data.dataset import EvalDataset
load_dotenv()

def get_default_log_path():
        """
        Returns the default log path based on the platform.
        This matches the logic in the Interceptor class.
        """
        import platform
        
        if platform.system() == "Darwin":  # macOS
            default_dir = os.path.expanduser("~/Documents/langchain_logs")
        elif platform.system() == "Windows":
            default_dir = os.path.join(os.path.expanduser("~"), "Documents", "langchain_logs")
        else:  # Other unix
            default_dir = os.path.expanduser("~/langchain_logs")
            
        return os.path.join(default_dir, "langchain_logs.json")


def parse_and_load():
    
    # Path to the logs json file
    log_path = get_default_log_path()

    # Initialize the OpenAI LLM
    api_key = os.environ["OPENAI_API_KEY"]
    openai = OpenAI(api_key=api_key)

    # Get the json string from the file path provided
    with open(log_path, "r") as f:
        log_document = json.load(f)

    messages = [
        {
            "role": "developer",
            "content": parse_langchain_logs_prompt
        }
    ]

    messages.append({
        "role": "user",
        "content": f"Langchain logs: {log_document}"
    })

    completion = openai.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
    )

    # Remove the ```json from the beginning and end of the string
    result = completion.choices[0].message.content.replace("```json", "").replace("```", "")

    # Convert the string to a json object
    result = json.loads(result)

    # Save the json object to a json file
    with open("langchain_data.json", "w") as f:
        json.dump(result, f, indent=4)
        
    # Create an eval dataset class from the json objectand return it
    dataset = {
        "questions": [d["user_prompt"] for d in result],
        "responses": [d["llm_response"] for d in result],
        "retrieved_contexts": [d["retrieved_contexts"] for d in result],
    }
    
    return EvalDataset(**dataset)