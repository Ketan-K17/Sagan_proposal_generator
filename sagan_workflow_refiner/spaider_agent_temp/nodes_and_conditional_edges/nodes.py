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

def research_query_generator(state: State) -> State:
    """
    Given a user prompt, this node determines if research is needed and generates appropriate queries.
    """
    print(f"{Fore.YELLOW}################ RESEARCH QUERY GENERATOR BEGIN #################")
    state["user_prompt"] = state["messages"][1].content
    try:
        # Add the query generator prompt to messages
        # research_query_generator_prompt = SystemMessage(
        #     content=RESEARCH_QUERY_GENERATOR_PROMPT.format(
        #         section_title=state.get("section_title"),
        #         section_text=state.get("section_text")
        #     )
        # )
        # state["messages"].append(research_query_generator_prompt)
        
        response = llm.invoke(state["messages"])

        if not response or not hasattr(response, 'content'):
            raise ValueError("Invalid response from LLM.")

        # Extract research queries from the response content
        response_content = json.loads(response.content)
        research_queries = response_content.get('research_queries', [])

        # Show the list of queries to the user
        print("Generated Research Queries:")
        for i, query in enumerate(research_queries, start=1):
            print(f"{i}. {query}")

        # Allow user to modify the list of queries
        user_input = input("Would you like to modify or add queries? (yes/no): ").strip().lower()
        if user_input == 'yes':
            modified_queries = []
            # First, handle existing queries
            for i, query in enumerate(research_queries, start=1):
                new_query = input(f"Modify query {i} (or press Enter to keep it unchanged): ").strip()
                if new_query:
                    modified_queries.append(new_query)
                else:
                    modified_queries.append(query)
            
            # Allow adding new queries
            while True:
                add_more = input("\nWould you like to add a new query? (yes/no): ").strip().lower()
                if add_more != 'yes':
                    break
                new_query = input(f"Enter new query {len(modified_queries) + 1}: ").strip()
                if new_query:
                    modified_queries.append(new_query)
            
            research_queries = modified_queries

        # do we need to query the db?
        research_needed = len(research_queries) > 0
        state["research_needed"] = research_needed
        state["research_queries"] = research_queries

        if not research_needed:
            print("\n\nNo research needed.\n\n")

        print(f"\n\n\n\nstate at the end of research query generator: \n")
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
                print(f"- {field_name.capitalize()}: {field_value}")
        print(f"################ RESEARCH QUERY GENERATOR END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ RESEARCH QUERY GENERATOR END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_title"] = None
        state["section_text"] = None
        state["rough_draft"] = None
        state["research_queries"] = None
        state["context"] = None
        return state

def research_query_answerer(state: State) -> State:
    """
    Takes the generated queries and executes them against the vector database.
    Only runs if research_needed is True.
    """
    print(f"{Fore.BLUE}################ RESEARCH QUERY ANSWERER BEGIN #################")

    try:
        if not state.get("research_needed"):
            state["context"] = []
            return state

        # Add the query answerer prompt to messages
        research_query_answerer_prompt = SystemMessage(
            content=RESEARCH_QUERY_ANSWERER_PROMPT.format(
                section_title=state.get("section_title"),
                section_text=state.get("section_text"),
                research_queries=state.get("research_queries")
            )
        )
        state["messages"].append(research_query_answerer_prompt)

        context = []
        for query in state["research_queries"]:
            result = query_chromadb(
                "C:\\Users\\ketan\\Desktop\\SPAIDER-SPACE\\sagan_workflow\\ingest_data\\mychroma_db",
                "sentence-transformers/all-MiniLM-L6-v2",
                query
            )
            context.extend(result)

        state["context"] = context

        print(f"\n\n\n\nstate at the end of query answerer: \n")
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
                print(f"- {field_name.capitalize()}: {field_value}")
        print(f"################ QUERY ANSWERER END #################{Style.RESET_ALL}")
        return state

    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"################ QUERY ANSWERER END #################{Style.RESET_ALL}")
        state["messages"] = [str(e)]
        state["section_title"] = None
        state["section_text"] = None
        state["rough_draft"] = None
        state["research_queries"] = None
        state["context"] = None
        return state

def formatter(state: State):
    print(f"{Fore.LIGHTGREEN_EX}################ FORMATTING NODE BEGIN #################")

    # Ensure state values are not None
    user_prompt = state.get("user_prompt", "")
    section_text = state.get("section_text", "")
    context = state.get("context", "")

    # populating the agent prompt
    formatter_prompt = FORMATTER_PROMPT.format(
        user_prompt=user_prompt,
        section_text=section_text,
        context=context
    )

    # appending the formatter prompt to list of messages
    state["messages"].append(SystemMessage(content=formatter_prompt))
    
    # invoking the llm, response in json.
    response = llm.invoke(state["messages"])
    print(f"RESPONSE: {response.content}")

    # Parse the response content directly, removing any potential JSON code block markers
    content = response.content
    if content.startswith('```json'):
        content = content[7:]  # Remove ```json
    if content.endswith('```'):
        content = content[:-3]  # Remove closing ```
    
    response_json = json.loads(content.strip())
    state["modified_section_text"] = response_json.get('modified_section_text')

    messages = state["messages"]
    if len(messages) >= 3:
        for message in messages[-3:]:
            print(f"{message.type}: {message.content}")
    else:
        for message in messages:
            print(f"{message.type}: {message.content}")
    for field_name, field_value in state.items():
        if field_name != "messages":
            print(f"- {field_name.capitalize()}: {field_value}")
    print(f"################ FORMATTING NODE END #################{Style.RESET_ALL}")
    return state

def human_input_node(state: State):
    print(f"{Fore.LIGHTMAGENTA_EX}################ HUMAN INPUT NODE BEGIN #################")
    response = input("Saves Changes to the section text? (yes/no): ")
    state["user_approval"] = response
    print(f"################ HUMAN INPUT NODE END #################{Style.RESET_ALL}")
    return state

def save_changes(state: State):
    print(f"{Fore.CYAN}################ SAVING CHANGES NODE BEGIN #################")
    try:
        # Read the entire file
        with open(state["rough_draft_path"], 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Remove any content before the first section
        if '# ' in content:
            main_content = content[content.find('# '):]
            prefix = content[:content.find('# ')]
        else:
            main_content = content
            prefix = ""
        
        # Split into sections
        sections = main_content.split('\n# ')
        if sections[0].startswith('# '):
            sections[0] = sections[0][2:]
        
        # Get the section number and validate it
        section_number = state["section_number"]
        if section_number < 1 or section_number > len(sections):
            raise ValueError(f"Section number {section_number} is out of range. File has {len(sections)} sections.")
        
        # Extract the title from the original section
        section = sections[section_number - 1]
        section_title = section.split('\n', 1)[0]
        
        # Create the new section with original title and modified text
        sections[section_number - 1] = f"{section_title}\n{state['modified_section_text']}"
        
        # Reconstruct the document
        new_content = prefix + '# ' + '\n# '.join(sections)
        
        # Write back to file
        with open(state["rough_draft_path"], 'w', encoding='utf-8') as file:
            file.write(new_content)
            
        print("Changes saved successfully!")
        
    except Exception as e:
        print(f"Error saving changes: {e}")
        
    print(f"################ SAVING CHANGES NODE END #################{Style.RESET_ALL}")
    return state

