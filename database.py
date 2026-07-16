import sqlite3

DATABASE_NAME = "kolam_analysis.db"


def create_database():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        image_name TEXT,

        pattern TEXT,

        dot_count INTEGER,

        contour_count INTEGER,

        symmetry TEXT,

        complexity REAL,

        confidence REAL,

        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    conn.commit()
    conn.close()


def insert_analysis(

        image_name,
        pattern,
        dot_count,
        contour_count,
        symmetry,
        complexity,
        confidence

):

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO analysis(

        image_name,
        pattern,
        dot_count,
        contour_count,
        symmetry,
        complexity,
        confidence

    )

    VALUES(?,?,?,?,?,?,?)

    """,(

        image_name,
        pattern,
        dot_count,
        contour_count,
        symmetry,
        complexity,
        confidence

    ))

    conn.commit()

    conn.close()


def get_history():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

    cursor.execute("""

    SELECT *

    FROM analysis

    ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return rows