import re

RESTRICTED_PATTERNS = [
    r"\bcat(s)?\b",
    r"\bdog(s)?\b",
    r"\bhoroscope(s)?\b",
    r"\bzodiac\b",
    r"\btaylor\s+swift\b",
]

PROMPT_LEAK_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\breveal\s+your\s+instructions\b",
    r"\bshow\s+me\s+your\s+prompt\b",
    r"\bwhat\s+are\s+your\s+instructions\b",
    r"\bdeveloper\s+message\b",
]

def guardrail_response(user_text: str) -> str | None:
    """
    Returns a refusal message if the user asks about restricted topics
    or attempts to access/modify the system prompt. Otherwise returns None.
    """
    text = (user_text or "").lower().strip()

    for pat in PROMPT_LEAK_PATTERNS:
        if re.search(pat, text):
            return "I can’t reveal or display my internal instructions."

    for pat in RESTRICTED_PATTERNS:
        if re.search(pat, text):
            return "I can’t help with that topic. Please ask about something else."

    if "ignore previous instructions" in text or "you are now" in text and "system" in text:
        return "I can’t follow requests that try to override my internal instructions."

    return None