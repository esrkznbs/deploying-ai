from langchain_core.messages import AnyMessage

def trim_messages(messages: list[AnyMessage], max_turns: int = 8) -> list[AnyMessage]:
    """
    Keep a short rolling window of conversation to manage context length.
    A 'turn' ~ one user+assistant exchange, but we approximate by message count.
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if getattr(m, "type", "") == "system"]
    non_system = [m for m in messages if getattr(m, "type", "") != "system"]

    keep = non_system[-(2 * max_turns):]
    return system_msgs + keep