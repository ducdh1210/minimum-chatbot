from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


class State(StateGraph):
    messages: Annotated[list, add_messages]


llm = init_chat_model("openai:gpt-4o-mini")


def chatbot(state: MessagesState):
    print(state["messages"])
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
# graph = graph_builder.compile(checkpointer=MemorySaver())
graph = graph_builder.compile()
config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
