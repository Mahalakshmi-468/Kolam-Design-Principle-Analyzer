import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "kolam_history.db"


def init_db():
    """Create the analysis_history table if it doesn't already exist."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            pattern TEXT,
            design_style TEXT,
            dot_count INTEGER,
            contour_count INTEGER,
            symmetry_level TEXT,
            similarity REAL,
            complexity REAL,
            grid_size TEXT,
            difficulty TEXT,
            confidence REAL,
            match_score REAL
        )
    """)

    conn.commit()
    conn.close()


def save_analysis(pattern, design_style, dot_count, contour_count,
                   symmetry_level, similarity, complexity, grid_size,
                   difficulty, confidence, match_score):
    """Insert a new analysis record and return its row id."""

    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO analysis_history (
            timestamp, pattern, design_style, dot_count, contour_count,
            symmetry_level, similarity, complexity, grid_size,
            difficulty, confidence, match_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pattern, design_style, dot_count, contour_count,
        symmetry_level, similarity, complexity, grid_size,
        difficulty, confidence, match_score
    ))

    conn.commit()
    row_id = cursor.lastrowid
    conn.close()

    return row_id


def get_history(limit=50):
    """Return the most recent analyses as a pandas DataFrame."""

    init_db()

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )

    conn.close()

    return df


def clear_history():
    """Delete all saved analysis records."""

    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()
