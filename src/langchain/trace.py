# Program to trace langchain logs using callbacks

import os
import platform
import json
from langchain.callbacks.base import BaseCallbackHandler

# Callback to trace langchain logs and save them to a json file.
class Interceptor(BaseCallbackHandler):
    """Logs all LangChain events just like `langchain.debug = True`."""
    
    def __init__(self, filename="langchain_logs.json"):
        self.filename = filename
        self.logs = []

    def log_event(self, event_type, data):
        """Logs events with timestamp and saves to memory."""
        self.logs.append({"event": event_type, "data": data})

    #LLM Events
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.log_event("llm_start", {"model": serialized.get("name"), "prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        generations = []
        for gen_list in response.generations:
            gen_items = []
            for gen in gen_list:
                gen_items.append({
                    "text": gen.text,
                    "generation_info": gen.generation_info
                })
            generations.append(gen_items)
        self.log_event("llm_end", {"response": generations})

    def on_llm_error(self, error, **kwargs):
        self.log_event("llm_error", {"error": str(error)})
        
    def on_retriever_start(self, serialized, query, **kwargs):
        self.log_event("retriever_start", {"retriever": serialized.get("name", "unknown"), "query": query})

    def on_retriever_end(self, documents, **kwargs):
        self.log_event("retriever_end", {"retrieved_docs": [doc.page_content for doc in documents]})
        
    def on_retriever_error(self, error, **kwargs):
        self.log_event("retriever_error", {"error": str(error)})
    
    #Chain Events
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.log_event("chain_start", {"chain": serialized["name"], "inputs": inputs})

    def on_chain_end(self, outputs, **kwargs):
        self.log_event("chain_end", {"outputs": outputs})

    def on_chain_error(self, error, **kwargs):
        self.log_event("chain_error", {"error": str(error)})

    #Tool Events
    def on_tool_start(self, serialized, inputs, **kwargs):
        self.log_event("tool_start", {"tool": serialized["name"], "inputs": inputs})

    def on_tool_end(self, output, **kwargs):
        self.log_event("tool_end", {"output": output})

    def on_tool_error(self, error, **kwargs):
        self.log_event("tool_error", {"error": str(error)})

    #Agent Events
    def on_agent_action(self, action, **kwargs):
        self.log_event("agent_action", {"tool": action.tool, "input": action.tool_input})

    def on_agent_finish(self, finish, **kwargs):
        self.log_event("agent_finish", {"output": finish.return_values})

    #Save Logs to JSON
    def log(self):
        
        # Define default log directories based on platform
        if platform.system() == "Darwin":  # macOS
            default_dir = os.path.expanduser("~/Documents/langchain_logs")
        elif platform.system() == "Windows":
            default_dir = os.path.join(os.path.expanduser("~"), "Documents", "langchain_logs")
        else:  # Other unix
            default_dir = os.path.expanduser("~/langchain_logs")
        
        # Create the directory if it doesn't exist
        os.makedirs(default_dir, exist_ok=True)
        
        if os.path.dirname(self.filename):
            log_path = self.filename
        else:
            log_path = os.path.join(default_dir, self.filename)
        
        print(f"Saving logs to: {log_path}")
        print(self.logs)
        
        with open(log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        
        return self.logs
    
interceptor = Interceptor()