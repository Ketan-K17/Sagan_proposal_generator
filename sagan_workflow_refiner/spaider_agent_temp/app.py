from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

# LOCAL IMPORTS.
from graph import create_graph, compile_graph, print_stream
from schemas import State
from prompts.prompts import RESEARCH_QUERY_GENERATOR_PROMPT


def extract_section(draft_path: str, section_number: int) -> tuple[str, str]:
    """
    Extract section title and text from a markdown file based on section number.
    Returns a tuple of (section_title, section_text).
    """
    with open(draft_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove any content before the first section
    if '# ' in content:
        content = content[content.find('# '):]
    
    # Split the content into sections based on level 1 headers
    sections = content.split('\n# ')
    if sections[0].startswith('# '):  # Handle first section if it starts with #
        sections[0] = sections[0][2:]
    
    # Ensure section number is valid
    if section_number < 1 or section_number > len(sections):
        raise ValueError(f"Section number {section_number} is out of range. File has {len(sections)} sections.")
    
    # Get the requested section
    section = sections[section_number - 1]
    
    # Split into title and content, handling subsections
    section_parts = section.split('\n', 1)
    section_title = section_parts[0].split(':', 1)[1].strip() if ':' in section_parts[0] else section_parts[0].strip()
    section_text = section_parts[1].strip() if len(section_parts) > 1 else ""
    
    return section_title, section_text

config = RunnableConfig(
    recursion_limit=50,
    configurable={"thread_id": "1"}
)
print(config)



if __name__ == "__main__":
    # creating graph workflow instance and then compiling it.
    verbose = True
    builder = create_graph()
    graph = compile_graph(builder)


    draft_path = "../input_draft/draft.md"
    section_number = 1
        
    s_title, s_text = extract_section(draft_path, section_number)

    # printing the graph.
    print(graph.get_graph().draw_mermaid())

    research_query_generator_prompt = RESEARCH_QUERY_GENERATOR_PROMPT.format(
        section_title=s_title,
        section_text=s_text
    )

    user_input = input("############# User: ")
    initial_input = {
        "messages": [SystemMessage(content=research_query_generator_prompt), HumanMessage(content=user_input)],
        "section_text": s_text,
        "section_title": s_title,
        "section_number": section_number,
        "rough_draft_path": draft_path
    }

    print_stream(graph.stream(initial_input, stream_mode="values", config=config))

    user_approval = input("Save changes to the section text? (yes/no): ")
    if user_approval.lower() == "yes":
        # If approved, continue the graph execution
        for event in graph.stream(None, config=config, stream_mode="values"):
            event['messages'][-1].pretty_print()
            
    else:
        print("Changes not saved.")

# Use the function to get section title and text
try:
    s_title, s_text = extract_section(draft_path, section_number)
except Exception as e:
    print(f"Error extracting section: {e}")
    s_title, s_text = "", ""