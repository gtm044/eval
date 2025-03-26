from typing import List, Dict, Any, Generator, Callable
import json
from langgraph.pregel.io import AddableValuesDict
import inspect

tracking_list = []

def track_variable(variable_name):
    """
    A decorator that tracks a specific variable in the decorated function
    and adds its value to a tracking list.
    
    Args:
        variable_name (str): The name of the variable to track
        
    Returns:
        function: The decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            frame = inspect.currentframe().f_back
            local_vars = frame.f_locals
            
            # Check if the variable exists in the local scope
            if variable_name in local_vars:
                tracking_list.append(local_vars[variable_name])
            elif isinstance(result, dict) and variable_name in result:
                tracking_list.append(result[variable_name])
            elif hasattr(result, variable_name):
                tracking_list.append(getattr(result, variable_name))
            elif isinstance(result, list) and len(result) > 0:
                tracking_list.extend(result)
            
            return result
        return wrapper
    return decorator


def message_to_dict(message):
    """Convert a LangChain Message object to a dictionary."""
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
                    # node_name is the name of the agent/tool which we have to append to the messages
                    if 'messages' in node_output:
                        all_messages.extend(node_output['messages'])
        # Handle cases where it might be another iterable
        elif hasattr(stream_result, '__iter__'):
            for step in stream_result:
                for node_name, node_output in step.items():
                    if 'messages' in node_output:
                        all_messages.extend(node_output['messages'])
    except Exception as e:
        print(f"Error processing stream: {e}")
    
    return {"messages": all_messages}


def log_lang(conversation: List[AddableValuesDict]) -> None:
    """
    Log the conversation to a JSON file.
    """
    serializable_results = []
    for result in conversation:
        # Extract the messages from the result
        messages = result["messages"]
        # Convert each message to a dictionary
        serializable_messages = [message_to_dict(message) for message in messages]
        serializable_results.append(serializable_messages)

    # Save to a JSON file
    with open("langgraph_results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)
    print("Results saved to langgraph_results.json")
    
    
def log_lang_stream() -> None:
    """
    Log the stream to a JSON file.
    """
    if not tracking_list:
        print("No streams have been tracked yet")
        return
        
    print(f"Processing {len(tracking_list)} tracked stream(s)")
    processed_results = []
    
    for idx, result in enumerate(tracking_list):
        try:
            processed_result = process_stream_to_invoke_format(result)
            processed_results.append(processed_result)
        except Exception as e:
            print(f"Error processing result {idx+1}: {e}")
        
    # Convert processed results to a JSON serializable format
    serializable_results = []
    for idx, result in enumerate(processed_results):
        try:
            # Extract the messages from the processed result
            messages = result["messages"]
            # Convert each message to a dictionary
            serializable_messages = [message_to_dict(message) for message in messages]
            serializable_results.append(serializable_messages)
        except Exception as e:
            print(f"Error serializing result {idx+1}: {e}")
        
    # Save to a JSON file
    with open("langgraph_stream_results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Results saved to langgraph_stream_results.json with {len(serializable_results)} conversations")
    
    # Clear tracking list after processing
    tracking_list.clear()
    
# Need to find a way to add the name field to each of the messages in the stream logs.
# Another issue is to add the initial human message in the stream logs in each list.
# This helps us to match the stream logs with the inital human message in the langgraph logs.