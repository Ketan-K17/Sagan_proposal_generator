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
    builder.add_node("research_query_generator", research_query_generator)
    builder.add_node("research_query_answerer", research_query_answerer)
    builder.add_node("formatter", formatter)
    builder.add_node("research_tools_node", research_tools_node)  

    # ADD EDGES TO THE GRAPH
    builder.add_edge(START, "research_query_generator")
    builder.add_edge("research_query_generator", "research_query_answerer")
    builder.add_conditional_edges(
        "research_query_answerer",
        research_tools_condition,
        {
            "research_tools_node": "research_tools_node",
            "formatter": "formatter"
        }
    )
    builder.add_edge("research_tools_node", "research_query_answerer")
    builder.add_edge("formatter", END)
    return builder

def compile_graph(builder):
    '''COMPILE GRAPH'''
    checkpointer = MemorySaver()
    graph = builder.compile(interrupt_after=["formatter"], checkpointer=checkpointer)
    return graph

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

