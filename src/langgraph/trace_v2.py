from typing import List, Dict, Any, Generator, Callable, Optional, Union, TypeVar, Generic, Iterable, Iterator
import json
import os
import platform
from pathlib import Path
import inspect
import traceback
import functools

T = TypeVar('T')

class TraceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceManager, cls).__new__(cls)
            cls._instance._tracking_list = []
            cls._instance._human_messages = []
        return cls._instance
    
    def add_tracking_data(self, data: Any) -> None:
        self._tracking_list.append(data)
    
    def add_human_message(self, message: Any) -> None:
        self._human_messages.append(message)
    
    def extend_tracking_data(self, data_list: List[Any]) -> None:
        self._tracking_list.extend(data_list)
    
    def get_tracking_data(self) -> List[Any]:
        return self._tracking_list
    
    def get_human_messages(self) -> List[Any]:
        return self._human_messages
        
    def clear_tracking_data(self) -> None:
        self._tracking_list.clear()
        self._human_messages.clear()
    
    @property
    def has_tracking_data(self) -> bool:
        return len(self._tracking_list) > 0
    
    @property
    def has_human_messages(self) -> bool:
        return len(self._human_messages) > 0


def get_default_save_directory():
    system = platform.system()
    if system == "Darwin":
        default_dir = os.path.expanduser("~/Documents/langchain_logs")
    elif system == "Windows":
        default_dir = os.path.join(os.path.expanduser("~"), "Documents", "langchain_logs")
    else:
        default_dir = os.path.expanduser("~/langchain_logs")
    os.makedirs(default_dir, exist_ok=True)
    return Path(default_dir)


class TracedAgent:
    def __init__(self, agent):
        self._agent = agent
        self._trace_manager = TraceManager()
        
    def invoke(self, input_data):
        self._extract_human_messages(input_data)
        response = self._agent.invoke(input_data)
        self._trace_manager.add_tracking_data(response)
        return response
    
    def stream(self, input_data):
        self._extract_human_messages(input_data)
        stream_gen = self._agent.stream(input_data)
        collected_steps = []
        
        def wrapped_generator():
            for step in stream_gen:
                collected_steps.append(step)
                yield step
            self._trace_manager.add_tracking_data(collected_steps)
            
        return wrapped_generator()
    
    def batch(self, inputs, *args, **kwargs):
        if not hasattr(self._agent, 'batch'):
            raise AttributeError("The wrapped agent does not support batch processing")
        
        for input_data in inputs:
            self._extract_human_messages(input_data)
        
        responses = self._agent.batch(inputs, *args, **kwargs)
        
        for response in responses:
            self._trace_manager.add_tracking_data(response)
        
        return responses
        
    def _extract_human_messages(self, input_data):
        try:
            if isinstance(input_data, str):
                self._trace_manager.add_human_message(input_data)
                return
                
            if isinstance(input_data, dict):
                if "messages" in input_data:
                    messages = input_data["messages"]
                    self._extract_from_messages(messages)
                elif "input" in input_data:
                    self._extract_human_messages(input_data["input"])
                elif "query" in input_data:
                    self._trace_manager.add_human_message(input_data["query"])
                elif "prompt" in input_data:
                    self._trace_manager.add_human_message(input_data["prompt"])
                return
                
            if isinstance(input_data, list):
                self._extract_from_messages(input_data)
                return
        except Exception as e:
            print(f"Error extracting human message: {str(e)}")
    
    def _extract_from_messages(self, messages):
        try:
            for message in messages:
                if isinstance(message, dict):
                    if message.get("type") == "human" or message.get("role") == "user":
                        if "content" in message:
                            self._trace_manager.add_human_message(message["content"])
                
                elif hasattr(message, "type") and hasattr(message, "content"):
                    if message.type == "human":
                        self._trace_manager.add_human_message(message.content)
                elif hasattr(message, "role") and hasattr(message, "content"):
                    if message.role == "user" or message.role == "human":
                        self._trace_manager.add_human_message(message.content)
                
                elif hasattr(message, "__class__") and "human" in message.__class__.__name__.lower():
                    if hasattr(message, "content"):
                        self._trace_manager.add_human_message(message.content)
        except Exception as e:
            print(f"Error extracting from messages: {str(e)}")


class TraceContext:
    def __init__(self, human_message):
        self.human_message = human_message
        self.trace_manager = TraceManager()
    
    def __enter__(self):
        self.trace_manager.add_human_message(self.human_message)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def as_message(self):
        return {"messages": [{"type": "human", "content": self.human_message}]}
    
    def as_string(self):
        return self.human_message
    
    def as_dict(self):
        return {"type": "human", "content": self.human_message}


def create_traced_agent(agent):
    return TracedAgent(agent)


def message_to_dict(message):
    try:
        if isinstance(message, dict):
            return message
            
        message_dict = {
            "type": message.type if hasattr(message, "type") else 
                   (message.role if hasattr(message, "role") else "unknown"),
            "content": message.content
        }
        if hasattr(message, 'tool_calls') and message.tool_calls:
            message_dict["tool_calls"] = []
            for tool_call in message.tool_calls:
                if isinstance(tool_call, dict):
                    message_dict["tool_calls"].append(tool_call)
                else:
                    message_dict["tool_calls"].append({
                        "name": tool_call.name,
                        "args": tool_call.args
                    })
        return message_dict
    except AttributeError as e:
        error_msg = f"Failed to convert message to dict. Message may not have expected attributes: {str(e)}"
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error converting message to dictionary: {str(e)}"
        raise RuntimeError(error_msg) from e


def process_stream_to_invoke_format(stream_result):
    all_messages = []
    try:
        if isinstance(stream_result, Generator):
            steps = list(stream_result)
        elif isinstance(stream_result, list):
            steps = stream_result
        else:
            steps = [stream_result]
        
        for step in steps:
            for node_name, node_output in step.items():
                if 'messages' in node_output:
                    all_messages.extend(node_output['messages'])
                    
    except TypeError as e:
        error_msg = f"Error processing stream: Stream result is not iterable or has unexpected format. {str(e)}"
        raise TypeError(error_msg) from e
    except KeyError as e:
        error_msg = f"Error processing stream: Expected key not found in node output. {str(e)}"
        raise KeyError(error_msg) from e
    except Exception as e:
        error_msg = f"Error processing stream: {str(e)}"
        raise RuntimeError(error_msg) from e
    
    return {"messages": all_messages}


def log_traces() -> Optional[str]:
    trace_manager = TraceManager()
    tracking_list = trace_manager.get_tracking_data()
    
    if not tracking_list:
        print("No streams have been tracked yet")
        return None
    print(f"Processing {len(tracking_list)} tracked stream(s)")
    processed_results = []
    human_messages = trace_manager.get_human_messages()
    print(f"Found {len(human_messages)} human messages")
    if human_messages:
        print("Human messages found:")
        for i, msg in enumerate(human_messages[:5]):
            print(f"  {i+1}: {msg[:50]}..." if len(msg) > 50 else f"  {i+1}: {msg}")
        if len(human_messages) > 5:
            print(f"  ...and {len(human_messages)-5} more")
            
    for idx, result in enumerate(tracking_list):
        try:
            processed_result = process_stream_to_invoke_format(result)
            
            if idx < len(human_messages):
                human_message = {
                    "type": "human", 
                    "content": human_messages[idx]
                }
                processed_result["messages"].insert(0, human_message)
                
            processed_results.append(processed_result)
        except Exception as e:
            error_msg = f"Error processing result {idx+1}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
    
    try:
        serializable_results = []
        for idx, result in enumerate(processed_results):
            try:
                messages = result["messages"]
                serializable_messages = [message_to_dict(message) for message in messages]
                serializable_results.append(serializable_messages)
            except Exception as e:
                error_msg = f"Error serializing result {idx+1}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
        
        save_dir = get_default_save_directory()
        file_path = save_dir / "langgraph_traces.json"
        
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        print(f"Results saved to {file_path} with {len(serializable_results)} conversations")
        trace_manager.clear_tracking_data()
        return str(file_path)
    except (IOError, PermissionError) as e:
        error_msg = f"Failed to write log file: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Error saving results: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        raise RuntimeError(error_msg) from e


def trace_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trace_manager = TraceManager()
        for arg in args:
            if isinstance(arg, str):
                trace_manager.add_human_message(arg)
                break
        result = func(*args, **kwargs)
        trace_manager.add_tracking_data(result)
        return result
    return wrapper


# Example usage:
# from langgraph.trace_improved import create_traced_agent, log_traces, TraceContext
#
# # Method 1: Wrap an existing agent
# agent = create_graph()  # Your LangGraph agent
# traced_agent = create_traced_agent(agent)
# response = traced_agent.invoke("What's the weather?")
# log_traces()  # Saves traces to file
#
# # Method 2: Use context manager
# with TraceContext("What's the capital of France?") as ctx:
#     response = agent.invoke(ctx.as_message())
# log_traces()  # Saves traces to file
#
# # Method 3: Backwards compatibility with decorator pattern
# @trace_call
# def ask_agent(query):
#     return agent.invoke({"messages": [{"type": "human", "content": query}]})
#
# ask_agent("How does photosynthesis work?")
# log_traces()  # Saves traces to file 