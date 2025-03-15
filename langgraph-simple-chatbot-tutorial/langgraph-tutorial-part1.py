from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

import os

ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

CHATBOT = "chatbot"

def chatbot(state):
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder.add_node(CHATBOT, chatbot)
graph_builder.set_entry_point(CHATBOT)
graph_builder.set_finish_point(CHATBOT)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q", "exit"]:
            print("Goodbye!!!")
            break

        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break