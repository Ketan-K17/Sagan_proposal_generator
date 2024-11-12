import json
import logging
from typing import List
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from colorama import init, Fore, Back, Style

'''LOCAL IMPORTS'''
from schemas import State
from prompts.prompts import *
from models.chatgroq import BuildChatGroq, BuildChatOpenAI

'''IMPORT ALL TOOLS HERE AND CREATE LIST OF TOOLS TO BE PASSED TO THE AGENT.'''
from tools.script_executor import run_script
from tools.file_tree import get_file_tree
# from tools.web_tool import web_search_tool
from tools.query_chromadb import query_chromadb
# from utils.mdtopdf import convert_md_to_pdf
from utils.latextopdf import latex_to_pdf

load_dotenv()
init()

terminal_tools = [run_script, get_file_tree]
research_tools = [query_chromadb]

'''LLM TO USE'''
# MODEL = "llama-3.1-70b-versatile"
# MODEL = "llama-3.1-8b-instant"
# MODEL = "gemma2-9b-it"
# MODEL = "llama3-groq-70b-8192-tool-use-preview"
# MODEL = "mixtral-8x7b-32768"
MODEL = "gpt-4o"
# llm = BuildChatGroq(model=MODEL, temperature=0)
llm = BuildChatOpenAI(model=MODEL, temperature=0)

llm_with_terminal_tools = llm.bind_tools(terminal_tools)
llm_with_research_tools = llm.bind_tools(research_tools)


research_tools_node = ToolNode(research_tools)

def researcher(state: State) -> State:
    """
    Given a user prompt, this node queries the vector database if it decides that more information is needed.
    """
    print(f"{Fore.YELLOW}################ RESEARCHER BEGIN #################")
    # system_prompt = SystemMessage(RESEARCHER_PROMPT.format(section_text=state["section_text"]))
    # state["messages"].append(system_prompt)

    try:
        response = llm_with_research_tools.invoke(state["messages"])
        # print(f"Response content: {response.content}")
        # print(f"Response type: {type(response)}")

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        # Extract research queries from the response content
        response_content = json.loads(response.content)
        research_queries = response_content.get('research_queries', [])

        print(f"Response content: {response.content}")
        print(f"Research Queries: {research_queries}")

        state["research_queries"] = research_queries

        print(f"\n\n\n\nstate at the end of researcher: \n")
        ## printing messages
        print("Messages: ")
        messages = state["messages"]
        if len(messages) >= 3:
            for message in messages[-3:]:
                print(f"{message.type}: {message.content}")
        else:
            for message in messages:
                print(f"{message.type}: {message.content}")

        ## printing fields other than messages.
        for field_name, field_value in state.items():
            if field_name != "messages":
                print(f"- {field_name}: {field_value}")
        print(f"################ RESEARCHER END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ RESEARCHER END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_title"] = None
        state["section_text"] = None
        state["rough_draft"] = None
        state["research_queries"] = None
        return state







