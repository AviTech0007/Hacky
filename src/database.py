
import sqlite3
import json
from pathlib import Path
from datetime import datetime


# ------------------------------------------------------------------
# Database location
# ------------------------------------------------------------------

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "users.db"
)

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


# ------------------------------------------------------------------
# Database initialization
# ------------------------------------------------------------------

def init_database():
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                settings TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        conn.commit()

    finally:
        conn.close()


# ------------------------------------------------------------------
# User management
# ------------------------------------------------------------------

def get_or_create_user(email):
    # Normalize email so different capitalization or
    # surrounding whitespace refers to the same account.
    email = email.strip().lower()

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,),
        )

        row = cursor.fetchone()

        if row:
            return row[0]

        cursor.execute(
            """
            INSERT INTO users (
                email,
                created_at
            )
            VALUES (?, ?)
            """,
            (
                email,
                datetime.utcnow().isoformat(),
            ),
        )

        conn.commit()

        return cursor.lastrowid

    finally:
        conn.close()


# ------------------------------------------------------------------
# Load user settings
# ------------------------------------------------------------------

def load_settings(user_id):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT settings
            FROM user_settings
            WHERE user_id = ?
            """,
            (user_id,),
        )

        row = cursor.fetchone()

        if row:
            return json.loads(row[0])

        return {}

    finally:
        conn.close()


# ------------------------------------------------------------------
# Save user settings
# ------------------------------------------------------------------

def save_settings(user_id, settings):
    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO user_settings (
                user_id,
                settings,
                updated_at
            )
            VALUES (?, ?, ?)

            ON CONFLICT(user_id)
            DO UPDATE SET
                settings = excluded.settings,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(settings),
                datetime.utcnow().isoformat(),
            ),
        )

        conn.commit()

    finally:
        conn.close()
