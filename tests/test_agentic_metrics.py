#!/usr/bin/env python3
"""
Test file for agentic metrics with diverse scenarios and edge cases.
Tests include various agent interactions with expected metric values.
"""

import json
from typing import List, Dict, Any
import numpy as np
from src.evaluator.metrics.agentic import (
    tool_call_accuracy,
    answer_correctness,
    answer_faithfulness,
    tool_accuracy
)

# Dictionary to store test cases and expected results
test_cases = {}
expected_results = {}

# -------------- WEB SEARCH AGENT SCENARIOS --------------

# Standard web search scenario
test_cases["web_search_basic"] = {
    "human_messages": ["What were the major announcements at Google I/O 2023?"],
    "ai_messages": [
        ["Based on the search results, major announcements at Google I/O 2023 included: PaLM 2 language model, AI features in Google Search called SGE, AI in Workspace, Bard improvements, new Pixel devices, and Android 14."]
    ],
    "tool_calls": [
        [{"name": "web_search", "args": {"query": "major announcements at Google I/O 2023"}}]
    ],
    "reference_tool_calls": [
        [{"name": "web_search", "args": {"query": "Google I/O 2023 major announcements"}}]
    ],
    "tool_outputs": [
        ["Google I/O 2023 featured several major announcements: PaLM 2 language model, SGE in Search, AI features in Workspace apps, Bard capabilities, Pixel Fold and Pixel 7a, and Android 14 preview."]
    ],
    "gt_tool_outputs": [
        ["Google I/O 2023 included: PaLM 2 model, SGE, Workspace AI features, Bard improvements, Pixel Fold and 7a, Android 14 beta."]
    ],
    "gt_answers": [
        "The major announcements at Google I/O 2023 included: PaLM 2 AI model, Search Generative Experience, AI features for Workspace, Bard improvements, new Pixel devices, and Android 14 updates."
    ]
}
expected_results["web_search_basic"] = {
    "tool_call_accuracy": 0.85,  # High similarity for semantic matching
    "answer_correctness": 0.90,  # Very similar answer to ground truth
    "answer_faithfulness": 3,  # Answer closely follows tool output
    "tool_accuracy": 0.92,       # Tool outputs match well
}

# Web search with completely different query but similar intent
test_cases["web_search_different_query"] = {
    "human_messages": ["Tell me about the latest Google developer conference highlights"],
    "ai_messages": [
        ["The latest Google developer conference (Google I/O 2023) featured announcements including PaLM 2 AI model, Search Generative Experience, AI features in Workspace, Bard improvements, new Pixel devices, and Android 14."]
    ],
    "tool_calls": [
        [{"name": "web_search", "args": {"query": "latest Google developer conference highlights 2023"}}]
    ],
    "reference_tool_calls": [
        [{"name": "web_search", "args": {"query": "Google I/O 2023 major announcements"}}]
    ],
    "tool_outputs": [
        ["Google I/O 2023 featured several major announcements: PaLM 2 language model, SGE in Search, AI features in Workspace apps, Bard capabilities, Pixel Fold and Pixel 7a, and Android 14 preview."]
    ],
    "gt_tool_outputs": [
        ["Google I/O 2023 included: PaLM 2 model, SGE, Workspace AI features, Bard improvements, Pixel Fold and 7a, Android 14 beta."]
    ],
    "gt_answers": [
        "The major announcements at Google I/O 2023 included: PaLM 2 AI model, Search Generative Experience, AI features for Workspace, Bard improvements, new Pixel devices, and Android 14 updates."
    ]
}
expected_results["web_search_different_query"] = {
    "tool_call_accuracy": 0.75,  # Different but related query
    "answer_correctness": 0.88,  # Similar answer content
    "answer_faithfulness": 3,  # Answer follows tool output
    "tool_accuracy": 0.92,       # Same tool outputs as before
}

# Web search with unfaithful answer
test_cases["web_search_unfaithful"] = {
    "human_messages": ["What were the major announcements at Google I/O 2023?"],
    "ai_messages": [
        ["Google I/O 2023 featured releases of Python 4.0, a new programming language called Dart++, and a quantum computing platform for consumers. They also announced the discontinuation of Android in favor of a new FuchsiaOS."]
    ],
    "tool_calls": [
        [{"name": "web_search", "args": {"query": "major announcements at Google I/O 2023"}}]
    ],
    "reference_tool_calls": [
        [{"name": "web_search", "args": {"query": "Google I/O 2023 major announcements"}}]
    ],
    "tool_outputs": [
        ["Google I/O 2023 featured several major announcements: PaLM 2 language model, SGE in Search, AI features in Workspace apps, Bard capabilities, Pixel Fold and Pixel 7a, and Android 14 preview."]
    ],
    "gt_tool_outputs": [
        ["Google I/O 2023 included: PaLM 2 model, SGE, Workspace AI features, Bard improvements, Pixel Fold and 7a, Android 14 beta."]
    ],
    "gt_answers": [
        "The major announcements at Google I/O 2023 included: PaLM 2 AI model, Search Generative Experience, AI features for Workspace, Bard improvements, new Pixel devices, and Android 14 updates."
    ]
}
expected_results["web_search_unfaithful"] = {
    "tool_call_accuracy": 0.85,  # Still good query match
    "answer_correctness": 0.15,  # Very incorrect answer
    "answer_faithfulness": 0,  # Answer doesn't follow tool output at all
    "tool_accuracy": 0.92,       # Tool outputs still match
}

# -------------- CODE ASSISTANT SCENARIOS --------------

# Code assistance with exact match tool call
test_cases["code_assistance_exact"] = {
    "human_messages": ["Fix my Python function: def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"],
    "ai_messages": [
        ["The function has an off-by-one error. Here's the fixed version with n=2 handled specially:\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        seq = [0, 1]\n        for i in range(2, n):\n            seq.append(seq[i-1] + seq[i-2])\n        return seq\n```"]
    ],
    "tool_calls": [
        [{"name": "code_analysis", "args": {"code": "def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"}}]
    ],
    "reference_tool_calls": [
        [{"name": "code_analysis", "args": {"code": "def fibonacci(n): if n <= 0: return []; elif n == 1: return [0]; else: seq = [0, 1]; for i in range(2, n): seq.append(seq[i-1] + seq[i-2]); return seq"}}]
    ],
    "tool_outputs": [
        ["The fibonacci function has an issue where it generates n-1 elements instead of n elements for n>1. Need to handle n=2 specially."]
    ],
    "gt_tool_outputs": [
        ["The fibonacci function has an issue where it generates n-1 elements instead of n elements for n>1. Need to handle n=2 specially."]
    ],
    "gt_answers": [
        "The function needs to handle n=2 specially:\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        seq = [0, 1]\n        for i in range(2, n):\n            seq.append(seq[i-1] + seq[i-2])\n        return seq\n```"
    ]
}
expected_results["code_assistance_exact"] = {
    "tool_call_accuracy": 1.0,   # Exact match
    "answer_correctness": 0.95,  # Very similar solution
    "answer_faithfulness": 3,  # Answer follows tool output
    "tool_accuracy": 1.0,        # Exact match on tool output
}

# Code assistance with similar but not identical call
test_cases["code_assistance_similar"] = {
    "human_messages": ["Debug this code: def fib(n): a,b=0,1; result=[]; for i in range(n): result.append(a); a,b=b,a+b; return result"],
    "ai_messages": [
        ["Your code has a subtle issue with indentation. Here's the fixed version:\n```python\ndef fib(n):\n    a, b = 0, 1\n    result = []\n    for i in range(n):\n        result.append(a)\n        a, b = b, a+b\n    return result\n```"]
    ],
    "tool_calls": [
        [{"name": "code_analysis", "args": {"code": "def fib(n): a,b=0,1; result=[]; for i in range(n): result.append(a); a,b=b,a+b; return result"}}]
    ],
    "reference_tool_calls": [
        [{"name": "analyze_code", "args": {"source_code": "def fib(n): a,b=0,1; result=[]; for i in range(n): result.append(a); a,b=b,a+b; return result"}}]
    ],
    "tool_outputs": [
        ["The code has indentation issues. The for loop body should include both the append and the variable reassignment."]
    ],
    "gt_tool_outputs": [
        ["This code has an indentation problem. Both result.append(a) and a,b=b,a+b should be in the for loop."]
    ],
    "gt_answers": [
        "The issue is with indentation. Both statements need to be part of the loop body:\n```python\ndef fib(n):\n    a, b = 0, 1\n    result = []\n    for i in range(n):\n        result.append(a)\n        a, b = b, a+b\n    return result\n```"
    ]
}
expected_results["code_assistance_similar"] = {
    "tool_call_accuracy": 0.50,  # Different name but same functionality
    "answer_correctness": 0.90,  # Very similar solution
    "answer_faithfulness": 2,  # Answer follows tool output idea
    "tool_accuracy": 0.85,       # Similar outputs but different wording
}

# -------------- DATA ANALYSIS SCENARIOS --------------

# Standard data analysis
test_cases["data_analysis_standard"] = {
    "human_messages": ["Analyze this sales data: {\"Product A\": {\"2022\": 1200000, \"2023\": 1450000}, \"Product B\": {\"2022\": 890000, \"2023\": 1230000}, \"Product C\": {\"2022\": 350000, \"2023\": 520000}}"],
    "ai_messages": [
        ["Product C had the highest percentage growth at 48.6%, increasing from $350,000 to $520,000. Product B had the second most growth of $340,000 (38.2% growth). Product A had the lowest growth rate at 20.8%."]
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
        ["Growth calculations: Product A: 20.83% ($250,000), Product B: 38.20% ($340,000), Product C: 48.57% ($170,000)"]
    ],
    "gt_answers": [
        "Product C had the highest percentage growth at 48.6%, while Product B had the highest absolute growth of $340,000. Product A had the lowest growth at 20.8%."
    ]
}
expected_results["data_analysis_standard"] = {
    "tool_call_accuracy": 0.75,  # Different parameter name
    "answer_correctness": 0.90,  # Very similar analysis
    "answer_faithfulness": 3,  # Accurately reflects tool output
    "tool_accuracy": 0.95,       # Very similar tool outputs
}

# -------------- EDGE CASES --------------

# Empty tool calls
test_cases["edge_empty_tool_calls"] = {
    "human_messages": ["What's the weather in New York?"],
    "ai_messages": [
        ["I don't have access to real-time weather data, but I can help you find weather information for New York."]
    ],
    "tool_calls": [
        []
    ],
    "reference_tool_calls": [
        [{"name": "weather", "args": {"location": "New York"}}]
    ],
    "tool_outputs": [
        []
    ],
    "gt_tool_outputs": [
        ["Weather in New York: 72°F, Partly Cloudy"]
    ],
    "gt_answers": [
        "Currently in New York, it's 72°F and partly cloudy."
    ]
}
expected_results["edge_empty_tool_calls"] = {
    "tool_call_accuracy": 0.0,   # No tool call when one was expected
    "answer_correctness": 0.30,  # Acknowledges lack of data
    "answer_faithfulness": 0,  # No tool output to be faithful to
    "tool_accuracy": 0.0,        # No tool output to compare
}

# Empty reference tool calls
test_cases["edge_empty_reference"] = {
    "human_messages": ["Tell me a joke"],
    "ai_messages": [
        ["Why don't scientists trust atoms? Because they make up everything!"]
    ],
    "tool_calls": [
        [{"name": "joke_generator", "args": {"category": "science"}}]
    ],
    "reference_tool_calls": [
        []
    ],
    "tool_outputs": [
        ["Why don't scientists trust atoms? Because they make up everything!"]
    ],
    "gt_tool_outputs": [
        []
    ],
    "gt_answers": [
        "Why don't scientists trust atoms? Because they make up everything!"
    ]
}
expected_results["edge_empty_reference"] = {
    "tool_call_accuracy": 0.0,   # Tool call when none expected
    "answer_correctness": 1.0,   # Same joke
    "answer_faithfulness": 3,  # Matches tool output
    "tool_accuracy": 0.0,        # No ground truth to compare
}

# Multiple sequential tool calls
test_cases["multiple_tool_calls"] = {
    "human_messages": ["What's the population of Tokyo and convert it to scientific notation"],
    "ai_messages": [
        ["The population of Tokyo is approximately 37.34 million people. In scientific notation, that's 3.734 × 10^7."]
    ],
    "tool_calls": [
        [
            {"name": "web_search", "args": {"query": "population of Tokyo"}},
            {"name": "calculate", "args": {"operation": "scientific_notation", "number": 37340000}}
        ]
    ],
    "reference_tool_calls": [
        [
            {"name": "web_search", "args": {"query": "Tokyo population"}},
            {"name": "convert", "args": {"value": 37340000, "format": "scientific_notation"}}
        ]
    ],
    "tool_outputs": [
        ["The population of Tokyo metropolitan area is approximately 37.34 million people (37,340,000).", 
         "3.734 × 10^7"]
    ],
    "gt_tool_outputs": [
        ["Tokyo has a population of approximately 37.34 million (37,340,000) as of 2021.",
         "3.734 × 10^7"]
    ],
    "gt_answers": [
        "Tokyo has a population of approximately 37.34 million. In scientific notation, that's 3.734 × 10^7."
    ]
}
expected_results["multiple_tool_calls"] = {
    "tool_call_accuracy": 0.65,  # Different tool names but similar args
    "answer_correctness": 0.95,  # Very similar answer
    "answer_faithfulness": 3,  # Follows both tool outputs
    "tool_accuracy": 0.90,       # Similar but not identical outputs
}

# Non-string arguments
test_cases["non_string_args"] = {
    "human_messages": ["Calculate 15% of 450"],
    "ai_messages": [
        ["15% of 450 is 67.5"]
    ],
    "tool_calls": [
        [{"name": "calculate", "args": {"percentage": 15, "value": 450}}]
    ],
    "reference_tool_calls": [
        [{"name": "calculate", "args": {"percentage": 15, "value": 450}}]
    ],
    "tool_outputs": [
        ["67.5"]
    ],
    "gt_tool_outputs": [
        ["67.5"]
    ],
    "gt_answers": [
        "15% of 450 is 67.5"
    ]
}
expected_results["non_string_args"] = {
    "tool_call_accuracy": 1.0,   # Exact match with numeric args
    "answer_correctness": 1.0,   # Exact match
    "answer_faithfulness": 3,  # Matches tool output
    "tool_accuracy": 1.0,        # Exact match
}

# Different tool ordering
test_cases["different_tool_ordering"] = {
    "human_messages": ["Convert 100 EUR to USD and calculate 15% tip"],
    "ai_messages": [
        ["100 EUR is approximately 109 USD. A 15% tip would be 16.35 USD, for a total of 125.35 USD."]
    ],
    "tool_calls": [
        [
            {"name": "currency_convert", "args": {"amount": 100, "from": "EUR", "to": "USD"}},
            {"name": "calculate", "args": {"percentage": 15, "value": 109}}
        ]
    ],
    "reference_tool_calls": [
        [
            {"name": "calculate", "args": {"percentage": 15, "value": 109}},
            {"name": "currency_convert", "args": {"amount": 100, "from": "EUR", "to": "USD"}}
        ]
    ],
    "tool_outputs": [
        ["100 EUR = 109 USD at current exchange rate", "15% of 109 is 16.35"]
    ],
    "gt_tool_outputs": [
        ["15% of 109 is 16.35", "100 EUR = 109 USD at current exchange rate"]
    ],
    "gt_answers": [
        "When converting 100 EUR to USD, you get approximately 109 USD. Adding a 15% tip of 16.35 USD gives a total of 125.35 USD."
    ]
}
expected_results["different_tool_ordering"] = {
    "tool_call_accuracy": 0.75,  # Same tools but different order
    "answer_correctness": 0.90,  # Similar content 
    "answer_faithfulness": 3,  # Follows tool outputs
    "tool_accuracy": 0.90,       # Same content in different order
}

# Completely incorrect tool call
test_cases["incorrect_tool"] = {
    "human_messages": ["What's the weather in London?"],
    "ai_messages": [
        ["The current exchange rate for 1 GBP is 1.25 USD."]
    ],
    "tool_calls": [
        [{"name": "currency_convert", "args": {"from": "GBP", "to": "USD", "amount": 1}}]
    ],
    "reference_tool_calls": [
        [{"name": "weather", "args": {"location": "London"}}]
    ],
    "tool_outputs": [
        ["1 GBP = 1.25 USD at current exchange rate"]
    ],
    "gt_tool_outputs": [
        ["Weather in London: 15°C, Rainy"]
    ],
    "gt_answers": [
        "Currently in London, it's 15°C and rainy."
    ]
}
expected_results["incorrect_tool"] = {
    "tool_call_accuracy": 0.0,   # Completely wrong tool
    "answer_correctness": 0.0,   # Unrelated answer
    "answer_faithfulness": 3,  # Faithful to wrong tool output
    "tool_accuracy": 0.0,        # Unrelated outputs
}

# Missing keys in arguments
test_cases["missing_keys"] = {
    "human_messages": ["Translate 'Hello' to French"],
    "ai_messages": [
        ["'Hello' in French is 'Bonjour'."]
    ],
    "tool_calls": [
        [{"name": "translate", "args": {"text": "Hello"}}]
    ],
    "reference_tool_calls": [
        [{"name": "translate", "args": {"text": "Hello", "source": "English", "target": "French"}}]
    ],
    "tool_outputs": [
        ["Bonjour"]
    ],
    "gt_tool_outputs": [
        ["The French translation of 'Hello' is 'Bonjour'"]
    ],
    "gt_answers": [
        "'Hello' translates to 'Bonjour' in French."
    ]
}
expected_results["missing_keys"] = {
    "tool_call_accuracy": 0.6,   # Missing keys
    "answer_correctness": 0.95,  # Almost identical answer
    "answer_faithfulness": 3,  # Follows tool output
    "tool_accuracy": 0.85,       # Similar output
}

# Long answer vs. short reference
test_cases["verbose_answer"] = {
    "human_messages": ["What's the capital of France?"],
    "ai_messages": [
        ["The capital of France is Paris, which is located in the north-central part of the country on the Seine River. Paris is one of the world's most important and attractive cities, known for its cultural heritage, fashion, gastronomy, and art. It's often called the 'City of Light' (la Ville Lumière) and is home to iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral."]
    ],
    "tool_calls": [
        [{"name": "knowledge", "args": {"query": "capital of France"}}]
    ],
    "reference_tool_calls": [
        [{"name": "knowledge", "args": {"query": "capital of France"}}]
    ],
    "tool_outputs": [
        ["The capital of France is Paris."]
    ],
    "gt_tool_outputs": [
        ["Paris is the capital of France."]
    ],
    "gt_answers": [
        "Paris is the capital of France."
    ]
}
expected_results["verbose_answer"] = {
    "tool_call_accuracy": 1.0,   # Exact match
    "answer_correctness": 0.75,  # Correct but overly verbose
    "answer_faithfulness": 2,  # Adds a lot not in tool output
    "tool_accuracy": 0.95,       # Very similar outputs
}

# -------------- GENERATE TESTS --------------

def run_metrics_tests():
    """Run all tests and compare with expected results."""
    print("Running tests for agentic metrics...")
    print("=" * 50)
    
    results = {}
    
    # Test all examples
    for test_name, test_case in test_cases.items():
        print(f"Testing: {test_name}")
        
        # Tool call accuracy
        tca_result = tool_call_accuracy(
            [test_case["tool_calls"][0]],
            [test_case["reference_tool_calls"][0]]
        )
        
        # Answer correctness
        ac_result = answer_correctness(
            [test_case["ai_messages"]],
            [test_case["gt_answers"][0]]
        )
        
        # Answer faithfulness
        af_result = answer_faithfulness(
            [test_case["ai_messages"]],
            [test_case["tool_outputs"]]
        )
        
        # Tool accuracy
        ta_result = tool_accuracy(
            [test_case["tool_outputs"]],
            [test_case["gt_tool_outputs"]]
        )
        
        # Store results
        results[test_name] = {
            "tool_call_accuracy": tca_result[0] if tca_result else 0.0,
            "answer_correctness": ac_result[0] if ac_result else 0.0,
            "answer_faithfulness": af_result[0] if af_result else 0.0,
            "tool_accuracy": ta_result[0] if ta_result else 0.0,
        }
        
        # Compare with expected
        expected = expected_results[test_name]
        print(f"  Tool Call Accuracy:  {results[test_name]['tool_call_accuracy']:.2f} (Expected: {expected['tool_call_accuracy']:.2f})")
        print(f"  Answer Correctness:  {results[test_name]['answer_correctness']:.2f} (Expected: {expected['answer_correctness']:.2f})")
        print(f"  Answer Faithfulness: {results[test_name]['answer_faithfulness']:.2f} (Expected: {expected['answer_faithfulness']:.2f})")
        print(f"  Tool Accuracy:       {results[test_name]['tool_accuracy']:.2f} (Expected: {expected['tool_accuracy']:.2f})")
        print("-" * 50)
    
    # Calculate average discrepancy
    discrepancies = []
    for test_name in results:
        metrics = ["tool_call_accuracy", "answer_correctness", "answer_faithfulness", "tool_accuracy"]
        for metric in metrics:
            discrepancy = abs(results[test_name][metric] - expected_results[test_name][metric])
            discrepancies.append(discrepancy)
    
    avg_discrepancy = np.mean(discrepancies)
    print(f"Average discrepancy from expected values: {avg_discrepancy:.4f}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    results = run_metrics_tests()
    
    # Write results to file
    with open("agentic_metrics_test_results.json", "w") as f:
        json.dump({
            "results": results,
            "expected": expected_results
        }, f, indent=2)
    
    print("Test results saved to agentic_metrics_test_results.json") 