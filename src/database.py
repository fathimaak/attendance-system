import sqlite3
import pickle
import numpy as np
from datetime import datetime
import os

DB_FILE = "data/attendance.db"
ENCODINGS_FILE = "data/encodings.pkl"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            registered_on TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized.")

def migrate_pickle_to_db():
    """One-time migration: move existing pickle data into SQLite."""
    if not os.path.exists(ENCODINGS_FILE):
        print("No existing pickle file to migrate.")
        return

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    for name, embedding in zip(data["names"], data["encodings"]):
        embedding_blob = pickle.dumps(embedding)  # store numpy array as blob
        c.execute(
            "INSERT INTO students (name, embedding, registered_on) VALUES (?, ?, ?)",
            (name, embedding_blob, datetime.now().isoformat())
        )

    conn.commit()
    conn.close()
    print(f"Migrated {len(data['names'])} student(s) into database.")

def add_student(name, embedding):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    embedding_blob = pickle.dumps(embedding)
    c.execute(
        "INSERT INTO students (name, embedding, registered_on) VALUES (?, ?, ?)",
        (name, embedding_blob, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_all_students():
    """Returns list of (name, embedding) tuples."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM students")
    rows = c.fetchall()
    conn.close()

    names = []
    embeddings = []
    for name, emb_blob in rows:
        names.append(name)
        embeddings.append(pickle.loads(emb_blob))

    return names, embeddings

def mark_attendance(student_name, confidence):
    """Mark attendance, but avoid duplicate marks within same day."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")
    c.execute(
        "SELECT * FROM attendance WHERE student_name = ? AND timestamp LIKE ?",
        (student_name, f"{today}%")
    )
    already_marked = c.fetchone()

    if already_marked:
        conn.close()
        return False  # already marked today

    c.execute(
        "INSERT INTO attendance (student_name, timestamp, confidence) VALUES (?, ?, ?)",
        (student_name, datetime.now().isoformat(), confidence)
    )
    conn.commit()
    conn.close()
    return True  # newly marked

def get_today_attendance():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute(
        "SELECT student_name, timestamp, confidence FROM attendance WHERE timestamp LIKE ?",
        (f"{today}%",)
    )
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    migrate_pickle_to_db()