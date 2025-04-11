import os
from dotenv import load_dotenv
import json
# from logger import log_lang_stream, log_lang
# from logger import track_variable
from src.langgraph.trace import log_lang_stream, track_variable
load_dotenv()

metal_price = {
    "gold": 88.1553,
    "silver": 1.0523,
    "platinum": 32.169,
    "palladium": 35.8252,
    "lbma_gold_am": 88.3294,
    "lbma_gold_pm": 88.2313,
    "lbma_silver": 1.0545,
    "lbma_platinum_am": 31.99,
    "lbma_platinum_pm": 32.2793,
    "lbma_palladium_am": 36.0088,
    "lbma_palladium_pm": 36.2017,
    "mcx_gold": 93.2689,
    "mcx_gold_am": 94.281,
    "mcx_gold_pm": 94.1764,
    "mcx_silver": 1.125,
    "mcx_silver_am": 1.1501,
    "mcx_silver_pm": 1.1483,
    "ibja_gold": 93.2713,
    "copper": 0.0098,
    "aluminum": 0.0026,
    "lead": 0.0021,
    "nickel": 0.0159,
    "zinc": 0.0031,
    "lme_copper": 0.0096,
    "lme_aluminum": 0.0026,
    "lme_lead": 0.002,
    "lme_nickel": 0.0158,
    "lme_zinc": 0.0031,
}


from langchain_core.tools import tool


# Define the tools for the agent to use
@tool
def get_metal_price(metal_name: str) -> float:
    """Fetches the current per gram price of the specified metal.

    Args:
        metal_name : The name of the metal (e.g., 'gold', 'silver', 'platinum').

    Returns:
        float: The current price of the metal in dollars per gram.

    Raises:
        KeyError: If the specified metal is not found in the data source.
    """
    try:
        metal_name = metal_name.lower().strip()
        if metal_name not in metal_price:
            raise KeyError(
                f"Metal '{metal_name}' not found. Available metals: {', '.join(metal_price['metals'].keys())}"
            )
        return metal_price[metal_name]
    except Exception as e:
        raise Exception(f"Error fetching metal price: {str(e)}")
    
    
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

tools = [get_metal_price]
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node transition    
def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Define the function that calls the model
def call_model(state: GraphState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node
def assistant(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

from langgraph.prebuilt import ToolNode

# Node
tools = [get_metal_price]
tool_node = ToolNode(tools)

from langgraph.graph import START, StateGraph
from IPython.display import Image, display

builder = StateGraph(GraphState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_continue, ["tools", END])
builder.add_edge("tools", "assistant")

react_graph = builder.compile()
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

from langchain_core.messages import HumanMessage

@track_variable("result_stream")
def get_agent_response(convo):
    all_streams = []
    for i in range(len(convo)):
        messages = [HumanMessage(content=convo[i])]
        result_stream = react_graph.stream({"messages": messages})
        all_streams.append(result_stream)
    return all_streams

get_agent_response(["What is the price of copper?", "What is the price of gold?"])

log_lang_stream()

from src.data.dataset import EvalDataset
from src.controller.options import ExperimentOptions
from src.controller.manager import Experiment

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

gt_dataset = EvalDataset(
    reference_tool_calls=reference_tool_calls,
    gt_answers=gt_answers,
    gt_tool_outputs=gt_tool_outputs
)

experiment_options = ExperimentOptions(
    langgraph=True
)

experiment = Experiment(dataset=gt_dataset, options=experiment_options)

# Try loading the experiment to couchbase
experiment.load_to_couchbase(collection="langgraph_test")