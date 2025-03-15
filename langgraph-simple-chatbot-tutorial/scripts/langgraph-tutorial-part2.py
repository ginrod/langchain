from typing import Annotated

from langchain.graphs import StateGraph, START, END
from langchain.graphs.message import add_messages

from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic

import os

ANTHROPIC_API_KEY = "ANTROPIC_API_KEY"
TAVILY_API_KEY = "TAVILY_API_KEY"

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

CHATBOT = "chatbot"

graph_builder.add_node(CHATBOT, chatbot)

import json

from langchain_core.messages import ToolMessage

class BasicToolNode:

    def __init__(self, tools):
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages in input")
        
        outputs = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        
        return {"messages": outputs}

TOOLS = "tools"

tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node(TOOLS, tool_node)

def route_tools(state):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
        return TOOLS
    return END

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    CHATBOT,
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    { TOOLS: TOOLS, END: END }
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge(TOOLS, CHATBOT)
graph_builder.set_entry_point(CHATBOT)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
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