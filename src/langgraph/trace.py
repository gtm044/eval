from typing import List, Dict, Any, Generator, Callable, Optional, Union
import json
import os
import platform
from pathlib import Path
from langgraph.pregel.io import AddableValuesDict
import inspect
import traceback


class TraceManager:
    """
    Singleton class for managing trace data and operations.
    Replaces the global tracking_list with a proper class-based implementation.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TraceManager, cls).__new__(cls)
            cls._instance._tracking_list = []
            cls._instance._human_messages = []
        return cls._instance
    
    def add_tracking_data(self, data: Any) -> None:
        """Add data to the tracking list"""
        self._tracking_list.append(data)
    
    def add_human_message(self, message: Any) -> None:
        """Add human message to the tracking"""
        self._human_messages.append(message)
    
    def extend_tracking_data(self, data_list: List[Any]) -> None:
        """Extend tracking list with multiple items"""
        self._tracking_list.extend(data_list)
    
    def get_tracking_data(self) -> List[Any]:
        """Get the current tracking data"""
        return self._tracking_list
    
    def get_human_messages(self) -> List[Any]:
        """Get the human messages"""
        return self._human_messages
        
    def clear_tracking_data(self) -> None:
        """Clear the tracking list"""
        self._tracking_list.clear()
        self._human_messages.clear()
    
    @property
    def has_tracking_data(self) -> bool:
        """Check if tracking list contains data"""
        return len(self._tracking_list) > 0


def get_default_save_directory():
    """
    Get the default directory for saving trace files based on the operating system.
    
    Returns:
        Path: The platform-specific default directory
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        default_dir = os.path.expanduser("~/Documents/langchain_logs")
    elif system == "Windows":
        default_dir = os.path.join(os.path.expanduser("~"), "Documents", "langchain_logs")
    else:  # Other unix
        default_dir = os.path.expanduser("~/langchain_logs")
    # Create directory if it doesn't exist
    os.makedirs(default_dir, exist_ok=True)
    return Path(default_dir)


def get_langgraph_logs():
    """
    Get the langgraph logs from the default save path
    """
    try:
        save_dir = get_default_save_directory()
        file_path = save_dir / "langgraph_stream_results.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found at {file_path}")
            
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON log file: {str(e)}"
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Error retrieving langgraph logs: {str(e)}"
        raise RuntimeError(error_msg) from e


def track_variable(variable_name, human_message=None, extract_human_messages=True):
    """
    A decorator that tracks a specific variable in the decorated function
    and adds its value to a tracking list. Optionally extracts human messages.
    
    Args:
        variable_name (str): The name of the variable to track
        extract_human_messages (bool): Whether to extract human messages from arguments/results
        
    Returns:
        function: The decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            trace_manager = TraceManager()
            
            # Special handling for conversation input parameters
            if extract_human_messages:
                # Get function signature to identify parameter names
                try:
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    
                    # Check for conversation parameters
                    for i, arg in enumerate(args):
                        if i < len(param_names) and param_names[i] in ['convo', 'conversation', 'messages', 'inputs']:
                            # This is likely a conversation parameter
                            if isinstance(arg, list):
                                for msg in arg:
                                    if isinstance(msg, str):
                                        trace_manager.add_human_message(msg)
                    
                    # Also check kwargs for conversation parameters
                    for key, value in kwargs.items():
                        if human_message is None and key in ['convo', 'conversation', 'messages', 'inputs']:
                            if isinstance(value, list):
                                for msg in value:
                                    if isinstance(msg, str):
                                        trace_manager.add_human_message(msg)
                        elif human_message in local_vars:
                            trace_manager.add_human_message(local_vars[human_message])
                            
                except Exception as e:
                    print(f"Error extracting conversations from function args: {e}")
            
            try:
                result = func(*args, **kwargs)
                frame = inspect.currentframe().f_back
                local_vars = frame.f_locals
                
                # Check if the variable exists in the local scope
                if variable_name in local_vars:
                    trace_manager.add_tracking_data(local_vars[variable_name])
                elif isinstance(result, dict) and variable_name in result:
                    trace_manager.add_tracking_data(result[variable_name])
                elif hasattr(result, variable_name):
                    trace_manager.add_tracking_data(getattr(result, variable_name))
                elif isinstance(result, list) and len(result) > 0:
                    trace_manager.extend_tracking_data(result)
                
                return result
            except Exception as e:
                error_msg = f"Error in tracking variable '{variable_name}': {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return func(*args, **kwargs)  # Fall back to original function
        return wrapper
    return decorator


def message_to_dict(message):
    """Convert a LangChain Message object to a dictionary."""
    try:
        # If message is already a dictionary, return it directly
        if isinstance(message, dict):
            return message
            
        message_dict = {
            "type": message.type,
            "content": message.content
        }
        # Add tool_calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            message_dict["tool_calls"] = []
            for tool_call in message.tool_calls:
                # Handle both object-style and dict-style tool calls
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
    """
    Process stream output from LangGraph and format it similar to invoke output.
    
    Args:
        stream_result: The stream result from LangGraph
    """
    all_messages = []
    try:
        # Handle cases where stream_result is a generator
        if isinstance(stream_result, Generator):
            # Consume the generator and collect all steps
            steps = list(stream_result)
            for step in steps:
                for node_name, node_output in step.items():
                    if 'messages' in node_output:
                        all_messages.extend(node_output['messages'])
                        
        # Handle cases where it might be another iterable
        elif hasattr(stream_result, '__iter__'):
            for step in stream_result:
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


def log_lang(conversation: List[AddableValuesDict]) -> str: # For single turn samples
    """
    Log the conversation to a JSON file in the default directory.
    
    Returns:
        str: Path to the saved JSON file
    """
    try:
        serializable_results = []
        for result in conversation:
            # Extract the messages from the result
            messages = result["messages"]
            # Convert each message to a dictionary
            serializable_messages = [message_to_dict(message) for message in messages]
            serializable_results.append(serializable_messages)

        # Get default save directory and create the file path
        save_dir = get_default_save_directory()
        file_path = save_dir / "langgraph_results.json"
        
        # Save to a JSON file
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Results saved to {file_path}")
        return str(file_path)
    except KeyError as e:
        error_msg = f"Failed to log conversation: Required key not found. {str(e)}"
        print(error_msg)
        raise KeyError(error_msg) from e
    except (IOError, PermissionError) as e:
        error_msg = f"Failed to write log file: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Error logging conversation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        raise RuntimeError(error_msg) from e
    
    
def log_lang_stream() -> Optional[str]: # Multi turn streams
    """
    Process tracked streams and save them to a JSON file in the default directory.
    
    Returns:
        str: Path to the saved JSON file, or None if no streams were tracked
    """
    trace_manager = TraceManager()
    tracking_list = trace_manager.get_tracking_data()
    
    if not tracking_list:
        print("No streams have been tracked yet")
        return None
        
    print(f"Processing {len(tracking_list)} tracked stream(s)")
    processed_results = []
    
    # Get human messages from trace manager
    human_messages = trace_manager.get_human_messages()
    print(f"Found {len(human_messages)} human messages")
    
    for idx, result in enumerate(tracking_list):
        try:
            processed_result = process_stream_to_invoke_format(result)
            
            # Add the corresponding human message at the beginning if available
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
        # Results -> JSON
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
        
        # Get default save directory
        save_dir = get_default_save_directory()
        file_path = save_dir / "langgraph_stream_results.json"
        
        # Save to a JSON file
        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        print(f"Results saved to {file_path} with {len(serializable_results)} conversations")
        
        # Clear tracking list after processing
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