"""Mock HR data for early development and tests."""

from __future__ import annotations

from typing import Any

MOCK_EMPLOYEES: dict[str, dict[str, Any]] = {
    "emp_001": {
        "id": "emp_001",
        "name": "Ahamed Shaa",
        "department": "Engineering",
        "manager": "Nadia Fernando",
        "leave_balance": 18,
        "schedule": "Monday-Friday, 9:00 AM-5:00 PM",
    },
    "emp_002": {
        "id": "emp_002",
        "name": "Maya Perera",
        "department": "People Operations",
        "manager": "Ruwan Jayasinghe",
        "leave_balance": 14,
        "schedule": "Monday-Friday, 8:30 AM-4:30 PM",
    },
    "emp_003": {
        "id": "emp_003",
        "name": "Daniel Kumar",
        "department": "Finance",
        "manager": "Maya Perera",
        "leave_balance": 21,
        "schedule": "Monday-Friday, 9:30 AM-5:30 PM",
    },
    "emp_004": {
        "id": "emp_004",
        "name": "Priya Silva",
        "department": "Customer Success",
        "manager": "Nadia Fernando",
        "leave_balance": 10,
        "schedule": "Tuesday-Saturday, 10:00 AM-6:00 PM",
    },
    "emp_005": {
        "id": "emp_005",
        "name": "Omar Rahman",
        "department": "Compliance",
        "manager": "Ruwan Jayasinghe",
        "leave_balance": 16,
        "schedule": "Monday-Friday, 8:00 AM-4:00 PM",
    },
}

MOCK_HR_POLICIES: list[dict[str, str]] = [
    {
        "name": "Leave Policy",
        "content": "Employees must request planned leave at least five business days in advance.",
    },
    {
        "name": "Scheduling Policy",
        "content": "Schedule changes require manager approval before the affected shift begins.",
    },
    {
        "name": "Compliance Rules",
        "content": "HR actions must be logged and handled according to company privacy guidelines.",
    },
]


def get_employee(user_id: str) -> dict[str, Any] | None:
    """Return a mock employee record by user ID."""
    return MOCK_EMPLOYEES.get(user_id)


def get_policies() -> list[dict[str, str]]:
    """Return all mock HR policy records."""
    return MOCK_HR_POLICIES
