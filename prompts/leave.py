"""
Prompts and response templates for the LeaveAgent.
"""

SYSTEM_PROMPT: str = (
    "You are an HR leave management assistant. "
    "Process leave requests by checking the employee's leave balance. "
    "Approve requests when the balance is sufficient. "
    "Deny requests politely when the balance is insufficient. "
    "Always state the outcome clearly and provide the updated balance."
)

# Placeholders: days_requested, new_balance
APPROVAL_TEMPLATE: str = (
    "Your leave request has been approved.\n"
    "Days approved: {days_requested}\n"
    "Remaining leave balance: {new_balance} days\n"
    "Please notify your manager and update your calendar."
)

# Placeholders: reason, current_balance
DENIAL_TEMPLATE: str = (
    "Unfortunately, your leave request cannot be approved.\n"
    "Reason: {reason}\n"
    "Current leave balance: {current_balance} days\n"
    "Please contact HR if you believe this is incorrect."
)

CLARIFICATION_TEMPLATE: str = (
    "To process your leave request, I need to know how many days you need "
    "and your preferred start date. Could you provide those details?"
)
