# Firstly we can evaluate with the langgraph logs alone. 
# How? -> keywords from questions and match it with the tool calls?

import json
import os
from typing import Dict, List, Optional, Any
from src.utils.models import openai_embedding
from src.utils.nlp import cosine_similarity
from src.evaluator.metrics.agentic import tool_call_accuracy, answer_correctness, answer_faithfulness, tool_accuracy


class AgentValidationEngine:
    """Validates agentic systems by comparing tool calls with references."""

    def __init__(
        self, 
        langgraph_logs: Optional[List[List[Dict[str, Any]]]],
        reference_tool_calls: Optional[List[List[Dict[str, Any]]]] = None,
        gt_answers: Optional[List[str]] = None,
        gt_tool_outputs: Optional[List[List[str]]] = None,
    ):
        """Initialize the validation engine.
        
        Args:
            langgraph_logs: Logs from langgraph execution
            reference_tool_calls: Reference tool calls for validation
            gt_answers: Ground truth answers for validation
            gt_tool_outputs: Ground truth tool outputs for validation
        """
        self.langgraph_logs = langgraph_logs
        self.reference_tool_calls = reference_tool_calls
        self.gt_answers = gt_answers
        self.gt_tool_outputs = gt_tool_outputs
        self.prepped_data = None
        self.metrics = [tool_call_accuracy, answer_correctness, answer_faithfulness, tool_accuracy]
        self.prep()

    def prep(self) -> Dict[str, Any]:
        """Evaluate agent performance by analyzing logs.
        
        Returns:
            Dictionary containing evaluation results
        """
        human_messages = []
        for conversation in self.langgraph_logs:
            for message in conversation:
                if message["type"] == "human":
                    human_messages.append(message["content"])
        
        ai_messages = []
        tool_calls = []
        for conversation in self.langgraph_logs:
            conv_ai = []
            for message in conversation:
                if "tool_calls" in message and message["tool_calls"]:
                    temp_tool_calls = []
                    for tool_call in message["tool_calls"]:
                        temp_tool_calls.append({
                            "name": tool_call["name"], 
                            "args": tool_call["args"]
                        })
                    tool_calls.append(temp_tool_calls)
                if message["type"] == "ai":
                    conv_ai.append(message["content"])
            ai_messages.append(conv_ai)
        
        tool_outputs = []
        for conversation in self.langgraph_logs:
            temp_tool_outputs = []
            for message in conversation:
                if message["type"] == "tool":
                    temp_tool_outputs.append(message["content"])
            tool_outputs.append(temp_tool_outputs)
        
        self.prepped_data = {
            "human_messages": human_messages,
            "ai_messages": ai_messages,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs
        }
        
        return self.prepped_data
        
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate agent performance by comparing tool calls with references.
        
        Returns:
            Dictionary containing evaluation results with metrics
        """
        if self.prepped_data is None:
            self.prep()
            
        results = {
            "tool_call_accuracy": [],
            "answer_correctness": [],
            "answer_faithfulness": [],
            "tool_accuracy": []
        }
        
        # Tool call accuracy - Compare agent's tool calls with reference tool calls
        if self.reference_tool_calls:
            results["tool_call_accuracy"] = tool_call_accuracy(self.prepped_data["tool_calls"], self.reference_tool_calls)
        
        # Answer correctness - Compare final AI responses with ground truth answers
        if self.gt_answers:
            results["answer_correctness"] = answer_correctness(self.prepped_data["ai_messages"], self.gt_answers)
        
        # Answer Faithfulness - Check if AI's answer is faithful to the tool outputs
        results["answer_faithfulness"] = answer_faithfulness(self.prepped_data["ai_messages"], self.prepped_data["tool_outputs"])
        
        # Tool Correctness - Check if the tool called is relevant to the user question
        # results["tool_correctness"] = tool_correctness(self.prepped_data["human_messages"], self.prepped_data["tool_calls"])
        
        # Tool Accuracy - Compare tool outputs with ground truth tool outputs
        if self.gt_tool_outputs:
            results["tool_accuracy"] = tool_accuracy(self.prepped_data["tool_outputs"], self.gt_tool_outputs)
        
        # The results should be processes as in @evaluator/validation.py
        results_json_list = []
        for data_point in zip(self.prepped_data["human_messages"], self.prepped_data["tool_calls"], self.prepped_data["ai_messages"], self.prepped_data["tool_outputs"], self.gt_answers, self.gt_tool_outputs, self.reference_tool_calls, results["tool_call_accuracy"], results["answer_correctness"], results["answer_faithfulness"], results["tool_accuracy"]):
            result_point = {
                "human_message": data_point[0],
                "tool_calls": data_point[1],
                "ai_messages": data_point[2],
                "tool_outputs": data_point[3],
                "ground_truth_answer": data_point[4],
                "ground_truth_tool_outputs": data_point[5],
                "reference_tool_calls": data_point[6],
                "tool_call_accuracy": data_point[7],
                "answer_correctness": data_point[8],
                "answer_faithfulness": data_point[9],
                "tool_accuracy": data_point[10]
            }
            results_json_list.append(result_point)
                
        avg_metrics = {}
        for metric in results.keys():
            if results[metric]:
                avg_metrics[metric] = sum(results[metric]) / len(results[metric])
            else:
                avg_metrics[metric] = None
                
        # Create a results directory
        results_dir = ".results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the results to a json file
        json_filename = os.path.join(results_dir, "results.json")
        with open(json_filename, "w") as f:
            json.dump(results_json_list, f, indent=4)
        
        return results_json_list, self.metrics, avg_metrics


if __name__ == "__main__":
    try:
        with open("/Users/goutham.krishnan/Documents/Codes/agent-test/langgraph_stream_results.json", "r") as f:
            langgraph_logs = json.load(f)
        
        # Example reference data (should be loaded from files in practice)
        reference_tool_calls = [
            [{"name": "get_metal_price", "args": {"metal_name": "copper"}}],
            [{"name": "get_metal_price", "args": {"metal_name": "gold"}}]
        ]
        
        gt_answers = [
            "The current price of copper is $0.0098 per gram.",
            "The current price of gold is $88.16 per gram."
        ]
        
        gt_tool_outputs = [
            ["0.0098"],
            ["88.1553"]
        ]
        
        engine = AgentValidationEngine(
            langgraph_logs=langgraph_logs, 
            reference_tool_calls=reference_tool_calls,
            gt_answers=gt_answers,
            gt_tool_outputs=gt_tool_outputs
        )
        
        prepped_data = engine.prep()
        eval_results, _, avg_metrics = engine.evaluate()
        
        # Save eval_results to a json file
        with open("eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)    
        
        # For debugging purposes only
        print("Prepped Data:")
        print(prepped_data)
        print("\nEvaluation Results:")
        print(eval_results)
        print("\nAverage Metrics:")
        print(avg_metrics)
        
    except FileNotFoundError:
        print("Error: Could not find langgraph logs file")
    except json.JSONDecodeError:
        print("Error: Could not parse langgraph logs file")