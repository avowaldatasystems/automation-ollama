from typing import TypedDict
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

import os

load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Embeddings
embeddings = OpenAIEmbeddings()

# Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Existing collection
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="pdf_docs",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

# State
class ChatState(TypedDict):
    question: str
    context: str
    answer: str

# Retrieve node
def retrieve(state: ChatState):

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {
        "context": context
    }

# Generate node
def generate_answer(state: ChatState):

    prompt = f"""
    Answer ONLY from the provided context.

    Context:
    {state['context']}

    Question:
    {state['question']}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }

# Build graph
graph = StateGraph(ChatState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()