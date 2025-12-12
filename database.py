"""
SQLite Database Module for Call Analytics and Recordings.

Provides persistent storage for:
- Call metadata and metrics
- Recording file references
- Historical analytics data

Replaces the JSON-based storage for better performance and queryability.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional



# CONFIGURATION

DEFAULT_DB_PATH = Path(__file__).parent / "call_analytics.db"


# DATABASE CONNECTION

@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Context manager for database connections."""
    path = db_path or str(DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database(db_path: Optional[str] = None):
    """Initialize the database schema."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Create calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0,
                avg_latency REAL DEFAULT 0,
                avg_bot_connection_time REAL DEFAULT 0,
                sentiment REAL DEFAULT 0,
                sentiment_label TEXT DEFAULT 'neutral',
                message_count INTEGER DEFAULT 0,
                avg_message_duration REAL DEFAULT 0,
                professionalism REAL DEFAULT 0,
                helpfulness REAL DEFAULT 0,
                engagement REAL DEFAULT 0,
                resolution REAL DEFAULT 0,
                speaker_sentiments TEXT DEFAULT '{}',
                customer_topics TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create recordings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id INTEGER,
                recording_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                sample_rate INTEGER DEFAULT 16000,
                num_channels INTEGER DEFAULT 1,
                duration_seconds REAL DEFAULT 0,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            )
        """)

        # Create topic_counts table for tracking topic frequency
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                count INTEGER DEFAULT 1
            )
        """)

        # Create aggregate_metrics table for historical averages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregate_metrics (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_calls INTEGER DEFAULT 0,
                all_calls_avg_latency REAL DEFAULT 0,
                all_calls_avg_bot_connection_time REAL DEFAULT 0,
                all_calls_avg_sentiment REAL DEFAULT 0,
                all_calls_total_duration REAL DEFAULT 0,
                all_calls_avg_duration REAL DEFAULT 0,
                all_calls_avg_message_duration REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize aggregate_metrics row if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO aggregate_metrics (id) VALUES (1)
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recordings_call_id ON recordings(call_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recordings_timestamp ON recordings(timestamp)")

        conn.commit()


# CALL OPERATIONS

def add_call(
    timestamp: str,
    duration_seconds: float,
    avg_latency: float,
    avg_bot_connection_time: float,
    sentiment: float,
    sentiment_label: str,
    message_count: int,
    avg_message_duration: float,
    professionalism: float = 0,
    helpfulness: float = 0,
    engagement: float = 0,
    resolution: float = 0,
    speaker_sentiments: Optional[Dict[str, float]] = None,
    customer_topics: Optional[List[str]] = None,
    db_path: Optional[str] = None
) -> int:
    """Add a new call record and update aggregate metrics."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Insert call record
        cursor.execute("""
            INSERT INTO calls (
                timestamp, duration_seconds, avg_latency, avg_bot_connection_time,
                sentiment, sentiment_label, message_count, avg_message_duration,
                professionalism, helpfulness, engagement, resolution,
                speaker_sentiments, customer_topics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            duration_seconds,
            avg_latency,
            avg_bot_connection_time,
            sentiment,
            sentiment_label,
            message_count,
            avg_message_duration,
            professionalism,
            helpfulness,
            engagement,
            resolution,
            json.dumps(speaker_sentiments or {}),
            json.dumps(customer_topics or [])
        ))
        call_id = cursor.lastrowid

        # Update topic counts
        if customer_topics:
            for topic in customer_topics:
                topic_lower = topic.lower().strip()
                cursor.execute("""
                    INSERT INTO topic_counts (topic, count)
                    VALUES (?, 1)
                    ON CONFLICT(topic) DO UPDATE SET count = count + 1
                """, (topic_lower,))

        # Update aggregate metrics
        cursor.execute("SELECT * FROM aggregate_metrics WHERE id = 1")
        agg = cursor.fetchone()

        n = agg["total_calls"]
        new_n = n + 1

        if n == 0:
            new_avg_latency = avg_latency
            new_avg_bot_conn = avg_bot_connection_time
            new_avg_sentiment = sentiment
            new_avg_duration = duration_seconds
            new_avg_msg_duration = avg_message_duration
        else:
            # Incremental average formula
            new_avg_latency = agg["all_calls_avg_latency"] + (avg_latency - agg["all_calls_avg_latency"]) / new_n
            new_avg_bot_conn = agg["all_calls_avg_bot_connection_time"] + (avg_bot_connection_time - agg["all_calls_avg_bot_connection_time"]) / new_n
            new_avg_sentiment = agg["all_calls_avg_sentiment"] + (sentiment - agg["all_calls_avg_sentiment"]) / new_n
            new_avg_duration = agg["all_calls_avg_duration"] + (duration_seconds - agg["all_calls_avg_duration"]) / new_n
            new_avg_msg_duration = agg["all_calls_avg_message_duration"] + (avg_message_duration - agg["all_calls_avg_message_duration"]) / new_n

        cursor.execute("""
            UPDATE aggregate_metrics SET
                total_calls = ?,
                all_calls_avg_latency = ?,
                all_calls_avg_bot_connection_time = ?,
                all_calls_avg_sentiment = ?,
                all_calls_total_duration = all_calls_total_duration + ?,
                all_calls_avg_duration = ?,
                all_calls_avg_message_duration = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            new_n,
            new_avg_latency,
            new_avg_bot_conn,
            new_avg_sentiment,
            duration_seconds,
            new_avg_duration,
            new_avg_msg_duration,
            datetime.now().isoformat()
        ))

        conn.commit()
        return call_id


# RECORDING OPERATIONS

def add_recording(
    recording_type: str,
    filename: str,
    filepath: str,
    timestamp: str,
    size_bytes: int = 0,
    sample_rate: int = 16000,
    num_channels: int = 1,
    duration_seconds: float = 0,
    call_id: Optional[int] = None,
    db_path: Optional[str] = None
) -> int:
    """Add a new recording record."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO recordings (
                call_id, recording_type, filename, filepath,
                size_bytes, sample_rate, num_channels, duration_seconds, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            call_id,
            recording_type,
            filename,
            filepath,
            size_bytes,
            sample_rate,
            num_channels,
            duration_seconds,
            timestamp
        ))

        conn.commit()
        return cursor.lastrowid


# QUERY OPERATIONS

def get_all_calls(limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all calls, newest first."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM calls
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        calls = []
        for row in cursor.fetchall():
            call = dict(row)
            call["speaker_sentiments"] = json.loads(call["speaker_sentiments"])
            call["customer_topics"] = json.loads(call["customer_topics"])
            calls.append(call)

        return calls


def get_call_history_for_dashboard(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Get call history data in the format expected by the dashboard."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Get aggregate metrics
        cursor.execute("SELECT * FROM aggregate_metrics WHERE id = 1")
        agg = cursor.fetchone()

        # Get all calls (limit to last 100)
        cursor.execute("""
            SELECT * FROM calls
            ORDER BY timestamp ASC
            LIMIT 100
        """)

        call_history = []
        for row in cursor.fetchall():
            call = {
                "timestamp": row["timestamp"],
                "duration_seconds": row["duration_seconds"],
                "avg_latency": row["avg_latency"],
                "avg_bot_connection_time": row["avg_bot_connection_time"],
                "sentiment": row["sentiment"],
                "sentiment_label": row["sentiment_label"],
                "message_count": row["message_count"],
                "avg_message_duration": row["avg_message_duration"],
                "professionalism": row["professionalism"],
                "helpfulness": row["helpfulness"],
                "engagement": row["engagement"],
                "resolution": row["resolution"],
                "speaker_sentiments": json.loads(row["speaker_sentiments"]),
                "customer_topics": json.loads(row["customer_topics"])
            }
            call_history.append(call)

        # Get topic counts
        cursor.execute("SELECT topic, count FROM topic_counts ORDER BY count DESC")
        topic_counts = {row["topic"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_calls": agg["total_calls"] if agg else 0,
            "all_calls_avg_latency": agg["all_calls_avg_latency"] if agg else 0,
            "all_calls_avg_bot_connection_time": agg["all_calls_avg_bot_connection_time"] if agg else 0,
            "all_calls_avg_sentiment": agg["all_calls_avg_sentiment"] if agg else 0,
            "all_calls_total_duration": agg["all_calls_total_duration"] if agg else 0,
            "all_calls_avg_duration": agg["all_calls_avg_duration"] if agg else 0,
            "all_calls_avg_message_duration": agg["all_calls_avg_message_duration"] if agg else 0,
            "call_history": call_history,
            "topic_counts": topic_counts
        }


def get_recordings(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all recordings, newest first."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM recordings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]


def get_recordings_grouped_by_session(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get recordings grouped by session timestamp."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM recordings
            ORDER BY timestamp DESC
        """)

        # Group by timestamp
        sessions = {}
        for row in cursor.fetchall():
            ts = row["timestamp"]
            if ts not in sessions:
                sessions[ts] = {
                    "timestamp": ts,
                    "datetime": None,
                    "files": {},
                    "call_id": row["call_id"]
                }
                # Parse datetime
                try:
                    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    sessions[ts]["datetime"] = dt
                except ValueError:
                    pass

            sessions[ts]["files"][row["recording_type"]] = {
                "id": row["id"],
                "path": row["filepath"],
                "size_kb": row["size_bytes"] / 1024,
                "name": row["filename"],
                "sample_rate": row["sample_rate"],
                "num_channels": row["num_channels"],
                "duration_seconds": row["duration_seconds"]
            }

        # Convert to sorted list (newest first)
        result = list(sessions.values())
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result


def link_recording_to_call(recording_id: int, call_id: int, db_path: Optional[str] = None):
    """Link a recording to a call."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE recordings SET call_id = ? WHERE id = ?
        """, (call_id, recording_id))
        conn.commit()


def get_latest_call_id(db_path: Optional[str] = None) -> Optional[int]:
    """Get the ID of the most recent call."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM calls ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        return row["id"] if row else None


# MIGRATION UTILITIES

def migrate_from_json(json_path: str, db_path: Optional[str] = None):
    """Migrate data from call_history.json to SQLite database."""
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    # Initialize database
    init_database(db_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    call_history = data.get("call_history", [])

    if not call_history:
        print("No call history to migrate")
        return

    print(f"Migrating {len(call_history)} calls from JSON to SQLite...")

    for call in call_history:
        add_call(
            timestamp=call.get("timestamp", datetime.now().isoformat()),
            duration_seconds=call.get("duration_seconds", 0),
            avg_latency=call.get("avg_latency", 0),
            avg_bot_connection_time=call.get("avg_bot_connection_time", 0),
            sentiment=call.get("sentiment", 0),
            sentiment_label=call.get("sentiment_label", "neutral"),
            message_count=call.get("message_count", 0),
            avg_message_duration=call.get("avg_message_duration", 0),
            professionalism=call.get("professionalism", 0),
            helpfulness=call.get("helpfulness", 0),
            engagement=call.get("engagement", 0),
            resolution=call.get("resolution", 0),
            speaker_sentiments=call.get("speaker_sentiments"),
            customer_topics=call.get("customer_topics"),
            db_path=db_path
        )

    print("Migration complete!")


if __name__ == "__main__":
    # Initialize the database
    print("Initializing database...")
    init_database()

    # Migrate from JSON if exists
    json_path = Path(__file__).parent / "call_history.json"
    if json_path.exists():
        migrate_from_json(str(json_path))
        print(f"Data migrated from {json_path}")
    else:
        print("No JSON file to migrate")

    # Verify
    data = get_call_history_for_dashboard()
    print(f"Database contains {data['total_calls']} calls")
