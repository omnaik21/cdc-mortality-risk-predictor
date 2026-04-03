"""
auth.py — Local Authentication System
CDC Mortality Risk Predictor | Group 6
Handles: register, login, password hashing, user storage (users.json)
"""

import json
import hashlib
import os
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(full_name: str, email: str, password: str, role: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    users = _load_users()

    email = email.strip().lower()

    if not full_name.strip():
        return False, "Full name cannot be empty."
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if email in users:
        return False, "An account with this email already exists."

    users[email] = {
        "full_name"  : full_name.strip(),
        "email"      : email,
        "password"   : _hash_password(password),
        "role"       : role,
        "created_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_users(users)
    return True, f"Account created successfully. Welcome, {full_name.split()[0]}!"


def login_user(email: str, password: str) -> tuple[bool, str, dict]:
    """Login a user. Returns (success, message, user_data)."""
    users = _load_users()
    email = email.strip().lower()

    if email not in users:
        return False, "No account found with this email.", {}
    if users[email]["password"] != _hash_password(password):
        return False, "Incorrect password. Please try again.", {}

    user = users[email]
    return True, f"Welcome back, {user['full_name'].split()[0]}!", user


def get_all_users() -> list:
    """Return list of all registered users (for admin view)."""
    users = _load_users()
    return [
        {k: v for k, v in u.items() if k != "password"}
        for u in users.values()
    ]
