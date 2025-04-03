## Metrics for agentic evaluation

# Tool Call Accuracy
# Answer Correctness
# Answer Faithfulness
# Tool Correctness
# Tool Accuracy

from typing import List, Dict, Any
from src.utils.models import openai_embedding
from src.utils.nlp import cosine_similarity

def tool_call_accuracy(tool_calls: List[List[Any]], reference_tool_calls: List[List[Dict[str, Any]]]) -> float:
    """Calculate tool call accuracy.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Tool call accuracy
    """
    tool_call_accuracy = []
    for i, (agent_calls, ref_calls) in enumerate(zip(tool_calls, reference_tool_calls)):
        conv_accuracy = 0.0
        if agent_calls and ref_calls:
            # Calculate accuracy based on matching tool names and arguments
            matches = 0
            for agent_call, ref_call in zip(agent_calls, ref_calls):
                if agent_call["name"] == ref_call["name"]:
                    # Basic match for tool name
                    name_match = 1.0
                    
                    # Check arguments match
                    args_match = 0.0
                    if agent_call["args"] == ref_call["args"]:
                        args_match = 1.0
                    elif isinstance(agent_call["args"], dict) and isinstance(ref_call["args"], dict):
                        # Partial match for arguments
                        common_keys = set(agent_call["args"].keys()) & set(ref_call["args"].keys())
                        matching_values = sum(1 for k in common_keys if agent_call["args"][k] == ref_call["args"][k])
                        args_match = matching_values / max(len(ref_call["args"]), 1)
                    
                    # Combined match score (50% for name, 50% for args)
                    matches += (name_match * 0.5 + args_match * 0.5)
            
            conv_accuracy = matches / max(len(ref_calls), 1)
    
        tool_call_accuracy.append(conv_accuracy)
    return tool_call_accuracy
    


def answer_correctness(ai_messages: List[List[Any]], gt_answers: List[str]) -> float:
    """Calculate answer correctness.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Answer correctness
    """
    answer_correctness = []
    for i, conv_ai_messages in enumerate(ai_messages):
        if conv_ai_messages and i < len(gt_answers):
            # Use the last AI message as the final answer
            final_answer = conv_ai_messages[-1]
            match_score = cosine_similarity(openai_embedding(final_answer), openai_embedding(gt_answers[i]))
            answer_correctness.append(round(match_score.item(), 2))
    return answer_correctness


def answer_faithfulness(ai_messages: List[List[Any]], tool_outputs: List[List[str]]) -> float:
    """Calculate answer faithfulness.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Answer faithfulness
    """
    answer_faithfulness = []
    for i, (conv_ai_messages, tool_outputs) in enumerate(zip(ai_messages, tool_outputs)):
        if conv_ai_messages and tool_outputs:
            # Simple check: does the final answer contain the tool output? -> bs, should use some kind of an llm based judge 
            final_answer = conv_ai_messages[-1]
            faithfulness = 0.0
            for output in tool_outputs:
                if output in final_answer:
                    faithfulness = 1.0
                    break
            answer_faithfulness.append(faithfulness)
    return answer_faithfulness


def tool_correctness(human_messages: List[str], tool_calls: List[List[Any]]) -> float:
    """Calculate tool correctness.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Tool correctness
    """
    tool_correctness = []
    for i, (human_message, tool_calls) in enumerate(zip(human_messages, tool_calls)):
        # Simplified implementation: always rate 1.0 if there are tool calls
        # In a real implementation, this would need NLP to check relevance
        correctness = 1.0 if tool_calls else 0.0
        tool_correctness.append(correctness)
    return tool_correctness


def tool_accuracy(tool_outputs: List[List[str]], gt_tool_outputs: List[List[str]]) -> float:
    """Calculate tool accuracy.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Tool accuracy
    """
    tool_accuracy = []
    for i, (tool_output, gt_output) in enumerate(zip(tool_outputs, gt_tool_outputs)):
        if tool_output and gt_output:
            matches = sum(1 for a, b in zip(tool_output, gt_output) if a == b)
            accuracy = matches / max(len(gt_output), 1)
            tool_accuracy.append(accuracy)
    return tool_accuracy


tool_call_accuracy.name = "tool_call_accuracy"
answer_correctness.name = "answer_correctness"
answer_faithfulness.name = "answer_faithfulness"
tool_correctness.name = "tool_correctness"
tool_accuracy.name = "tool_accuracy"