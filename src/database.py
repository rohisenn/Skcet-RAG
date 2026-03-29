import sqlite3
import datetime

DB_FILE = "skcet_rag.db"

def init_db():
    """Initializes the SQLite database and creates/migrates the queries table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            rating TEXT DEFAULT 'none',
            response_time_ms INTEGER DEFAULT 0,
            confidence TEXT DEFAULT 'unknown',
            is_flagged INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migration: add columns if they don't exist yet (for existing databases)
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(queries)")]
    if "response_time_ms" not in existing_cols:
        cursor.execute("ALTER TABLE queries ADD COLUMN response_time_ms INTEGER DEFAULT 0")
    if "confidence" not in existing_cols:
        cursor.execute("ALTER TABLE queries ADD COLUMN confidence TEXT DEFAULT 'unknown'")
    if "is_flagged" not in existing_cols:
        cursor.execute("ALTER TABLE queries ADD COLUMN is_flagged INTEGER DEFAULT 0")
    conn.commit()
    conn.close()

def flag_query(query_id):
    """Marks a query as flagged for review by an admin."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('UPDATE queries SET is_flagged = 1 WHERE id = ?', (query_id,))
    conn.commit()
    conn.close()

def log_query(user_input, assistant_response, response_time_ms=0, confidence="unknown"):
    """Logs a new query and response to the database. Returns the generated ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO queries (user_input, assistant_response, response_time_ms, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_input, assistant_response, response_time_ms, confidence, timestamp))
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id

def update_rating(query_id, rating):
    """Updates the rating (thumbs_up or thumbs_down) for a specific query."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('UPDATE queries SET rating = ? WHERE id = ?', (rating, query_id))
    conn.commit()
    conn.close()

def get_all_queries():
    """Retrieves all queries for the admin dashboard."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM queries ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_analytics():
    """Calculates analytics from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM queries')
    total_queries = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries WHERE rating = 'thumbs_up'")
    thumbs_up = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries WHERE rating = 'thumbs_down'")
    thumbs_down = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(response_time_ms) FROM queries WHERE response_time_ms > 0")
    avg_time = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as count
        FROM queries
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 14
    """)
    daily_rows = cursor.fetchall()

    cursor.execute("""
        SELECT DATE(timestamp) as day, AVG(response_time_ms) as avg_ms
        FROM queries
        WHERE response_time_ms > 0
        GROUP BY DATE(timestamp)
        ORDER BY day DESC
        LIMIT 14
    """)
    daily_perf = cursor.fetchall()

    conn.close()
    return {
        "total_queries": total_queries,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "avg_response_time_ms": round(avg_time),
        "daily_counts": [{"day": r[0], "count": r[1]} for r in reversed(daily_rows)],
        "daily_perf": [{"day": r[0], "avg_ms": round(r[1])} for r in reversed(daily_perf)],
    }
