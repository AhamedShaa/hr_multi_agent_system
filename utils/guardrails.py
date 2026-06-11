"""
guardrails.py — Input validation for domain relevance and prompt injection.

Called before the orchestrator pipeline runs. Prevents:
1. Off-topic requests (non-HR messages)
2. Prompt injection attempts (instructions embedded in user message)
"""

from __future__ import annotations

HR_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "leave", "vacation", "sick", "time off", "time-off", "absence",
    "schedule", "meeting", "interview", "calendar", "appointment", "booking",
    "policy", "compliance", "regulation", "procedure", "rule", "hr",
    "human resources", "payroll", "benefits", "employment", "contract",
    "manager", "department", "team", "colleague", "office", "work",
    "annual leave", "parental", "maternity", "paternity", "days off",
})

INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "disregard all",
    "forget your instructions",
    "new instructions:",
    "override instructions",
    "ignore your system prompt",
    "you are now",
    "pretend you are",
    "act as if",
    "jailbreak",
    "dan mode",
    "developer mode",
)

OUT_OF_DOMAIN_RESPONSE: str = (
    "I'm an HR assistant and can only help with leave requests, "
    "scheduling, or HR policy questions. "
    "Is there an HR topic I can help you with today?"
)

INJECTION_RESPONSE: str = (
    "I'm unable to process that request as written. "
    "If you have an HR question, please rephrase it clearly."
)


def is_hr_related(message: str) -> bool:
    """Return True if the message appears to be HR-related."""
    lower_message = message.lower()
    return any(keyword in lower_message for keyword in HR_DOMAIN_KEYWORDS)


def has_injection_attempt(message: str) -> bool:
    """Return True if the message contains known prompt injection patterns."""
    lower_message = message.lower()
    return any(pattern in lower_message for pattern in INJECTION_PATTERNS)


def validate_input(message: str) -> tuple[bool, str, str]:
    """
    Validate user message for domain relevance and injection attempts.

    Args:
        message: The raw user message string.

    Returns:
        Tuple of (is_valid: bool, rejection_reason: str, response_message: str).
        If is_valid is True, rejection_reason and response_message are empty strings.
    """
    if has_injection_attempt(message):
        return False, "injection_attempt", INJECTION_RESPONSE
    if not is_hr_related(message):
        return False, "out_of_domain", OUT_OF_DOMAIN_RESPONSE
    return True, "", ""
