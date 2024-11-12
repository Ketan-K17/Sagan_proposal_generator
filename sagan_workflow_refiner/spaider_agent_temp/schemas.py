from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()

class State(MessagesState):
    section_title: str
    section_text: str
    rough_draft: str
    research_queries: list[str]
