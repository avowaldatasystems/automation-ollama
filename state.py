from typing_extensions import TypedDict
from typing import List


class AgentState(TypedDict):

    question: str
    response: str
    chat_history: List[dict]