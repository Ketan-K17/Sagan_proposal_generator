from langgraph.graph import StateGraph, END, START
from langgraph.constants import END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

# LOCAL IMPORTS
from nodes_and_conditional_edges.nodes import *
from nodes_and_conditional_edges.conditional_edges import *
from tools import *
from schemas import State


load_dotenv()

def create_graph():
    # GRAPH INSTANCE
    builder = StateGraph(State)

    # ADD NODES TO THE GRAPH
    builder.add_node("researcher", researcher)
    builder.add_node("researcher_toolnode", research_tools_node)  

    # ADD EDGES TO THE GRAPH
    builder.add_edge(START, "researcher")
    builder.add_conditional_edges(
        "researcher",
        researcher_tools_condition,
        {
            "researcher_toolnode": "researcher_toolnode",
            "__end__": END
        }
    )
    builder.add_edge("researcher_toolnode", "researcher")

    return builder

def compile_graph(builder):
    '''COMPILE GRAPH'''
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

