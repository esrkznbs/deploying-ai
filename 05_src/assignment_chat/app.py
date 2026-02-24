
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

from main import get_assignment_chat_agent


agent = get_assignment_chat_agent()


def assignment_chat(message, history):

    langchain_messages = []

    for user_msg, bot_msg in history:
        if user_msg:
            langchain_messages.append(HumanMessage(content=user_msg))
        if bot_msg:
            langchain_messages.append(AIMessage(content=bot_msg))

    langchain_messages.append(HumanMessage(content=message))

    state = {"messages": langchain_messages}
    result = agent.invoke(state)

    for msg in reversed(result["messages"]):
        if getattr(msg, "type", "") == "ai":
            return msg.content

    return "No response generated."


demo = gr.ChatInterface(
    assignment_chat,
    title="Assignment Chat (InsightBot)",
    description="Try: 'Give me a quote' | 'semantic search: moral panic' | 'analyze text: <paste text>'",
)

demo.launch()