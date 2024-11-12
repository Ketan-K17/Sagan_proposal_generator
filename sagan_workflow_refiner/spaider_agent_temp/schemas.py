from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()

class State(MessagesState):
    user_prompt: str
    word_limit: int
    section_text: str
    modified_section_text: str
    research_needed: bool
    section_title: str
    rough_draft: str
    research_queries: list[str]
    context: list[str] # answers to the queries from the vector database
