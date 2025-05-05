## Metrics for agentic evaluation

# Tool Call Accuracy
# Answer Correctness
# Answer Faithfulness
# Tool Correctness
# Tool Accuracy

from typing import List, Dict, Any
from src.utils.models import openai_embedding
from src.utils.nlp import cosine_similarity
from src.utils.models import llm_as_a_judge

def tool_call_accuracy(agent_tool_calls: List[List[Any]], reference_tool_calls: List[List[Dict[str, Any]]]) -> float:
    """Calculate tool call accuracy with more precise targeting of semantic matching."""
    
    # Define tool-specific semantic matching rules
    # Format: {tool_name: {param_name: should_use_semantic}}
    SEMANTIC_RULES = {
        "web_search": {"query": True},
        "search": {"query": True, "search_term": True},
        "ask": {"question": True, "prompt": True},
        "summarize": {"text": True, "content": True},
        # Add other tools as needed
        
        # Default for all other tools - exact matching for all params
        "*": {"*": False}
    }
    
    # Minimum length for natural language to use semantic matching
    MIN_LENGTH = 10
    
    tool_call_accuracy = []
    for i, (agent_calls, ref_calls) in enumerate(zip(agent_tool_calls, reference_tool_calls)):
        conv_accuracy = 0.0
        if agent_calls and ref_calls:
            matches = 0
            for agent_call, ref_call in zip(agent_calls, ref_calls):
                if agent_call["name"] == ref_call["name"]:
                    # Basic match for tool name
                    name_match = 1.0
                    tool_name = agent_call["name"]
                    
                    # Check arguments match with tool-specific rules
                    args_match = 0.0
                    if agent_call["args"] == ref_call["args"]:
                        args_match = 1.0
                    elif isinstance(agent_call["args"], dict) and isinstance(ref_call["args"], dict):
                        common_keys = set(agent_call["args"].keys()) & set(ref_call["args"].keys())
                        matching_values = 0
                        
                        # Get rules for this tool, fallback to default
                        tool_rules = SEMANTIC_RULES.get(tool_name, SEMANTIC_RULES["*"])
                        
                        for k in common_keys:
                            # Check if this parameter should use semantic matching
                            use_semantic = tool_rules.get(k, tool_rules.get("*", False))
                            
                            # Apply semantic matching only if explicitly configured
                            if (use_semantic and 
                                isinstance(agent_call["args"][k], str) and 
                                isinstance(ref_call["args"][k], str) and 
                                len(agent_call["args"][k]) >= MIN_LENGTH and
                                len(ref_call["args"][k]) >= MIN_LENGTH):
                                
                                agent_embedding = openai_embedding(agent_call["args"][k])
                                ref_embedding = openai_embedding(ref_call["args"][k])
                                similarity = cosine_similarity(agent_embedding, ref_embedding)
                                matching_values += similarity
                            # Use exact matching for everything else
                            elif agent_call["args"][k] == ref_call["args"][k]:
                                matching_values += 1
                                
                        args_match = matching_values / max(len(ref_call["args"]), 1)
                
                    # Combined match score
                    matches += (name_match * 0.5 + args_match * 0.5)
            
            conv_accuracy = matches / max(len(ref_calls), 1)
        
        tool_call_accuracy.append(round(conv_accuracy if not hasattr(conv_accuracy, 'item') else conv_accuracy.item(), 2))
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

## Couple of new metrics to be added:
# 1. One for comparing the agent responses (n number of responses) with the gt agent responses
# 2. One for comparing the agent responses with the question and the tool outputs to get the overall_target_score or something like that

def agent_response_correctness(agent_responses: List[List[Any]], gt_agent_responses: List[List[Any]]) -> float:
    """Calculate agent response correctness -> semantic similarity between agent responses and gt agent responses and penalty for message length mismatch.
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs
    Returns:
        float: Agent response correctness
    """
    conversation_scores = []
    for agent_conv, gt_conv in zip(agent_responses, gt_agent_responses):
        turn_scores = []
        num_turns_match = min(len(agent_conv), len(gt_conv))
        if num_turns_match == 0:
            conversation_scores.append(0.0)
            continue
        for i in range(num_turns_match):
            agent_msg = str(agent_conv[i])
            gt_msg = str(gt_conv[i])
            if not agent_msg or not gt_msg:
                turn_scores.append(0.0)
                continue
            # agent_embedding = openai_embedding(agent_msg)
            # gt_embedding = openai_embedding(gt_msg)
            # similarity = cosine_similarity(agent_embedding, gt_embedding)
            # turn_scores.append(round(similarity.item(), 2))
            # Use BERTScore instead of cosine similarity
            from bert_score import score
            P, R, F1 = score([agent_msg], [gt_msg], lang="en", verbose=False)
            # Use F1 score as the similarity measure
            similarity = F1.item()
            turn_scores.append(round(similarity, 2))
        conversation_scores.append(sum(turn_scores) / num_turns_match)
    return conversation_scores
    

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
            faithfulness = llm_as_a_judge(final_answer, tool_outputs)
            answer_faithfulness.append(faithfulness)
    return answer_faithfulness


### NO SENSE TO INCLUDE THIS METRIC AS OF NOW
# def tool_correctness(human_messages: List[str], tool_calls: List[List[Any]]) -> float:
#     """Calculate tool correctness.
    
#     Args:
#         langgraph_logs: List of lists of dictionaries containing conversation logs

#     Returns:
#         float: Tool correctness
#     """
#     tool_correctness = []
#     for i, (human_message, tool_calls) in enumerate(zip(human_messages, tool_calls)):
#         # Simplified implementation: always rate 1.0 if there are tool calls
#         # In a real implementation, this would need NLP to check relevance
#         correctness = 1.0 if tool_calls else 0.0
#         tool_correctness.append(correctness)
#     return tool_correctness


def tool_accuracy(agent_tool_outputs: List[List[str]], gt_tool_outputs: List[List[str]]) -> float:
    """Calculate tool accuracy.
    
    Args:
        langgraph_logs: List of lists of dictionaries containing conversation logs

    Returns:
        float: Tool accuracy
    """
    tool_accuracy = []
    for i, (agent_tool_output, gt_tool_output) in enumerate(zip(agent_tool_outputs, gt_tool_outputs)):
        # Basic string matching implementationnas of now, need to find something more relevant
        #Cosine similarity -> bw the ground truth tool output and the tool output
        if agent_tool_output and gt_tool_output:
            # Calculate cosine similarity between tool outputs and ground truth
            similarities = []
            for a, b in zip(agent_tool_output, gt_tool_output):
                a_str = str(a)
                b_str = str(b)
                
                a_embedding = openai_embedding(a_str)
                b_embedding = openai_embedding(b_str)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(a_embedding, b_embedding)
                similarities.append(similarity)
            
            # Average similarity across all tool outputs
            accuracy = sum(similarities) / max(len(similarities), 1)
            tool_accuracy.append(round(accuracy.item(), 2))
    return tool_accuracy


tool_call_accuracy.name = "tool_call_accuracy"
answer_correctness.name = "answer_correctness"
answer_faithfulness.name = "answer_faithfulness"
tool_accuracy.name = "tool_accuracy"
agent_response_correctness.name = "agent_response_correctness"

if __name__=='__main__':
    # Example data for testing metrics
    # Scenario 1: Web search agent
    web_search_example = {
        "human_messages": ["What were the major announcements at Google I/O 2023?"],
        "ai_messages": [
            ["I'll search for information about Google I/O 2023 announcements.", 
             "Based on the search results, major announcements at Google I/O 2023 included: 1) PaLM 2 language model, 2) AI features in Google Search called SGE, 3) AI features in Workspace apps, 4) Improved Bard capabilities, 5) New Pixel devices including Pixel Fold and Pixel 7a, and 6) Android 14 features."]
        ],
        "tool_calls": [
            [{"name": "web_search", "args": {"query": "major announcements at Google I/O 2023"}}]
        ],
        "reference_tool_calls": [
            [{"name": "web_search", "args": {"query": "Google I/O 2023 major announcements"}}]
        ],
        "tool_outputs": [
            ["Google I/O 2023 featured several major announcements: PaLM 2 language model powering Bard and other AI products; SGE (Search Generative Experience) bringing AI to Google Search; AI features in Gmail, Docs, and other Workspace apps; upgraded Bard capabilities; Pixel Fold and Pixel 7a launches; and Android 14 preview."]
        ],
        "gt_tool_outputs": [
            ["Google I/O 2023 included these key announcements: 1) PaLM 2 language model, 2) Search Generative Experience (SGE), 3) Workspace AI features for Gmail and Docs, 4) Bard improvements, 5) New hardware including Pixel Fold and Pixel 7a, 6) Android 14 beta."]
        ],
        "gt_answers": [
            "The major announcements at Google I/O 2023 included: PaLM 2 AI model, AI-powered Search Generative Experience, AI features for Google Workspace, improvements to Bard, new Pixel devices (Fold and 7a), and Android 14 updates."
        ]
    }
    
    # Scenario 2: Code assistance agent
    code_assistance_example = {
        "human_messages": ["Help me fix this Python function that's supposed to calculate the Fibonacci sequence but has bugs: def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"],
        "ai_messages": [
            ["I'll analyze this function for you.", 
             "The function has an off-by-one error. Here's the fixed version:\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    else:\n        seq = [0, 1]\n        for i in range(2, n):\n            seq.append(seq[i-1] + seq[i-2])\n        return seq\n```\nThis will correctly return a Fibonacci sequence of length n."]
        ],
        "tool_calls": [
            [{"name": "code_analysis", "args": {"code": "def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"}}]
        ],
        "reference_tool_calls": [
            [{"name": "code_analysis", "args": {"code": "def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"}}]
        ],
        "tool_outputs": [
            ["The function has these issues: 1) The function doesn't handle n=2 specially but should return [0,1], 2) For n>2, it correctly builds the sequence but the loop should run to n, not n-1, to get n elements."]
        ],
        "gt_tool_outputs": [
            ["Code analysis: The fibonacci function has an issue where it generates n-1 elements instead of n elements for n>1. The loop should run for range(2, n) which gives n-2 iterations after already having 2 elements [0,1]."]
        ],
        "gt_answers": [
            "The function has an issue where it generates n-1 Fibonacci numbers instead of n. To fix it:\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        seq = [0, 1]\n        for i in range(2, n):\n            seq.append(seq[i-1] + seq[i-2])\n        return seq\n```"
        ]
    }
    
    # Scenario 3: Data analysis agent
    data_analysis_example = {
        "human_messages": ["Analyze this sales data and tell me which product had the highest growth: {\"Product A\": {\"2022\": 1200000, \"2023\": 1450000}, \"Product B\": {\"2022\": 890000, \"2023\": 1230000}, \"Product C\": {\"2022\": 350000, \"2023\": 520000}}"],
        "ai_messages": [
            ["I'll analyze this sales data for you.",
             "Based on the data, Product B had the highest growth rate at 38.2%, followed by Product C at 48.6%, and Product A at 20.8%. In absolute terms, Product B increased by $340,000, Product A by $250,000, and Product C by $170,000."]
        ],
        "tool_calls": [
            [{"name": "calculate", "args": {"operation": "growth_percentage", "data": {"Product A": {"2022": 1200000, "2023": 1450000}, "Product B": {"2022": 890000, "2023": 1230000}, "Product C": {"2022": 350000, "2023": 520000}}}}]
        ],
        "reference_tool_calls": [
            [{"name": "calculate", "args": {"operation": "growth_rate", "data": {"Product A": {"2022": 1200000, "2023": 1450000}, "Product B": {"2022": 890000, "2023": 1230000}, "Product C": {"2022": 350000, "2023": 520000}}}}]
        ],
        "tool_outputs": [
            ["Growth results: Product A: 20.8% ($250,000), Product B: 38.2% ($340,000), Product C: 48.6% ($170,000)"]
        ],
        "gt_tool_outputs": [
            ["Growth calculations: Product A: 20.83% growth ($250,000 increase), Product B: 38.20% growth ($340,000 increase), Product C: 48.57% growth ($170,000 increase)"]
        ],
        "gt_answers": [
            "Product C had the highest percentage growth at 48.6%, increasing from $350,000 in 2022 to $520,000 in 2023. Product B had the highest absolute growth of $340,000, rising from $890,000 to $1,230,000 (38.2% growth). Product A had the lowest growth rate at 20.8%."
        ]
    }
    
    # Combine all examples
    examples = [web_search_example, code_assistance_example, data_analysis_example]
    
    # Test all metrics with the examples
    print("Testing agentic metrics with realistic examples...")
    
    # Tool call accuracy
    tca_results = tool_call_accuracy(
        [ex["tool_calls"][0] for ex in examples],
        [ex["reference_tool_calls"][0] for ex in examples]
    )
    print(f"Tool Call Accuracy: {tca_results}")
    
    # Answer correctness
    ac_results = answer_correctness(
        [ex["ai_messages"] for ex in examples],
        [ex["gt_answers"][0] for ex in examples]
    )
    print(f"Answer Correctness: {ac_results}")
    
    # Answer faithfulness
    af_results = answer_faithfulness(
        [ex["ai_messages"] for ex in examples],
        [ex["tool_outputs"] for ex in examples]
    )
    print(f"Answer Faithfulness: {af_results}")
    
    # Tool accuracy
    ta_results = tool_accuracy(
        [ex["tool_outputs"] for ex in examples],
        [ex["gt_tool_outputs"] for ex in examples]
    )
    print(f"Tool Accuracy: {ta_results}")