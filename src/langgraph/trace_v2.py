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
    """
    Singleton class for managing trace data and operations.
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
    
    @property
    def has_human_messages(self) -> bool:
        """Check if human messages list contains data"""
        return len(self._human_messages) > 0


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


class TracedAgent:
    """
    Wrapper for LangGraph agents that automatically traces input/output messages.
    
    This class allows for transparent tracing of agent invocations without
    modifying the original agent code. It works with both single-turn and
    streaming interfaces.
    """
    
    def __init__(self, agent):
        """
        Initialize a traced agent wrapper.
        
        Args:
            agent: The LangGraph agent to wrap
        """
        self._agent = agent
        self._trace_manager = TraceManager()
        
    def invoke(self, input_data):
        """
        Traced version of agent.invoke() that captures human messages and responses.
        
        Args:
            input_data: The input data for the agent (can be a string, dict, or list)
            
        Returns:
            The response from the wrapped agent
        """
        # Extract and track human messages from input
        self._extract_human_messages(input_data)
        
        # Call the original agent
        response = self._agent.invoke(input_data)
        
        # Store the response in tracking data
        self._trace_manager.add_tracking_data(response)
        
        return response
    
    def stream(self, input_data):
        """
        Traced version of agent.stream() that captures human messages and stream results.
        
        Args:
            input_data: The input data for the agent (can be a string, dict, or list)
            
        Returns:
            A generator yielding the same items as the original stream
        """
        # Extract and track human messages from input
        self._extract_human_messages(input_data)
        
        # Get the original stream generator
        stream_gen = self._agent.stream(input_data)
        
        # Collect all steps to track them while preserving streaming behavior
        collected_steps = []
        
        # Return a wrapped generator that tracks responses
        def wrapped_generator():
            for step in stream_gen:
                collected_steps.append(step)
                yield step
            
            # After streaming is complete, add the collected steps to tracking
            self._trace_manager.add_tracking_data(collected_steps)
            
        return wrapped_generator()
    
    def batch(self, inputs, *args, **kwargs):
        """
        Traced version of agent.batch() if the agent supports it.
        
        Args:
            inputs: List of inputs for batch processing
            *args, **kwargs: Additional arguments for the batch method
            
        Returns:
            The batch responses from the wrapped agent
        """
        if not hasattr(self._agent, 'batch'):
            raise AttributeError("The wrapped agent does not support batch processing")
        
        # Extract and track human messages from each input
        for input_data in inputs:
            self._extract_human_messages(input_data)
        
        # Call the original batch method
        responses = self._agent.batch(inputs, *args, **kwargs)
        
        # Store each response in tracking data
        for response in responses:
            self._trace_manager.add_tracking_data(response)
        
        return responses
        
    def _extract_human_messages(self, input_data):
        """
        Extract human messages from input data in various formats.
        
        Args:
            input_data: The input data to extract messages from
        """
        try:
            # Handle string input (direct human message)
            if isinstance(input_data, str):
                self._trace_manager.add_human_message(input_data)
                return
                
            # Handle dict with messages key
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
                
            # Handle list of messages directly
            if isinstance(input_data, list):
                self._extract_from_messages(input_data)
                return
        except Exception as e:
            print(f"Error extracting human message: {str(e)}")
    
    def _extract_from_messages(self, messages):
        """
        Extract human messages from a list of message objects.
        
        Args:
            messages: List of message objects
        """
        try:
            for message in messages:
                # Handle dict-style messages
                if isinstance(message, dict):
                    if message.get("type") == "human" or message.get("role") == "user":
                        if "content" in message:
                            self._trace_manager.add_human_message(message["content"])
                
                # Handle object-style messages
                elif hasattr(message, "type") and hasattr(message, "content"):
                    if message.type == "human":
                        self._trace_manager.add_human_message(message.content)
                elif hasattr(message, "role") and hasattr(message, "content"):
                    if message.role == "user" or message.role == "human":
                        self._trace_manager.add_human_message(message.content)
                
                # Handle special message classes
                elif hasattr(message, "__class__") and "human" in message.__class__.__name__.lower():
                    if hasattr(message, "content"):
                        self._trace_manager.add_human_message(message.content)
        except Exception as e:
            print(f"Error extracting from messages: {str(e)}")


class TraceContext:
    """
    Context manager for tracing conversations with explicit human messages.
    
    Example:
        with TraceContext("What is the capital of France?") as ctx:
            response = agent.invoke(ctx.as_message())
    """
    
    def __init__(self, human_message):
        """
        Initialize the trace context with a human message.
        
        Args:
            human_message: The human message to trace
        """
        self.human_message = human_message
        self.trace_manager = TraceManager()
    
    def __enter__(self):
        # Register the human message
        self.trace_manager.add_human_message(self.human_message)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to clean up
        pass
    
    def as_message(self):
        """Convert to a message format suitable for agent input"""
        return {"messages": [{"type": "human", "content": self.human_message}]}
    
    def as_string(self):
        """Return the original human message string"""
        return self.human_message
    
    def as_dict(self):
        """Return as a dictionary format"""
        return {"type": "human", "content": self.human_message}


def create_traced_agent(agent):
    """
    Factory function to create a traced version of a LangGraph agent.
    
    Args:
        agent: The agent to wrap with tracing
        
    Returns:
        A TracedAgent instance wrapping the provided agent
    """
    return TracedAgent(agent)


def message_to_dict(message):
    """Convert a LangChain Message object to a dictionary."""
    try:
        # If message is already a dictionary, return it directly
        if isinstance(message, dict):
            return message
            
        message_dict = {
            "type": message.type if hasattr(message, "type") else 
                   (message.role if hasattr(message, "role") else "unknown"),
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
            
        # Use the collected steps
        elif isinstance(stream_result, list):
            steps = stream_result
        else:
            steps = [stream_result]
        
        # Process each step
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
    
    # Debug information - useful to see what's being captured
    if human_messages:
        print("Human messages found:")
        for i, msg in enumerate(human_messages[:5]):  # Print first 5 for debugging
            print(f"  {i+1}: {msg[:50]}..." if len(msg) > 50 else f"  {i+1}: {msg}")
        if len(human_messages) > 5:
            print(f"  ...and {len(human_messages)-5} more")
    
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
        file_path = save_dir / "langgraph_traces.json"
        
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


def trace_call(func):
    """
    Decorator to trace function calls with automatic message extraction.
    For backwards compatibility with the original trace.py approach.
    
    Args:
        func: The function to trace
        
    Returns:
        The wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trace_manager = TraceManager()
        
        # Try to extract human messages from arguments
        for arg in args:
            if isinstance(arg, str):
                trace_manager.add_human_message(arg)
                break
                
        # Call the original function
        result = func(*args, **kwargs)
        
        # Add the result to tracking data
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