"""
Prompts and response templates for the SchedulingAgent.
"""

SYSTEM_PROMPT: str = (
    "You are an HR scheduling assistant. "
    "Help employees book meetings, interviews, and manage calendar requests. "
    "Always confirm the date, time, and attendees before finalizing. "
    "Be concise and professional."
)

# Placeholders: scheduled_time, calendar_id
SCHEDULED_TEMPLATE: str = (
    "Your meeting has been scheduled.\n"
    "Date and time: {scheduled_time}\n"
    "Calendar reference: {calendar_id}\n"
    "You will receive a calendar invite shortly."
)

NO_DATE_TEMPLATE: str = (
    "To schedule this, I need a few more details.\n"
    "Please provide: the preferred date, the preferred time, "
    "and the names of any attendees."
)

TIMEOUT_RESPONSE: str = (
    "Scheduling request timed out. Please try again or contact HR directly."
)
