from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

import os

ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

TAVILY_API_KEY = "TAVILY_API_KEY"

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

CHATBOT = "chatbot"
TOOLS = "tools"

graph_builder.add_node(CHATBOT, chatbot)

tool_node = ToolNode(tools)
graph_builder.add_node(TOOLS, tool_node)

graph_builder.add_conditional_edges(CHATBOT, tools_condition)

graph_builder.add_edge(TOOLS, CHATBOT)
graph_builder.set_entry_point(CHATBOT)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable":{"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="values"):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "q", "exit"]:
            print("Bye, bye")
            break
    
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break