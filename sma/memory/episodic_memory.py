import sqlite3
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

class EpisodicMemory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            
            # Sessions Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                summary TEXT
            );
            """)
            
            # Trajectories Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                yaw REAL NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """)

            # Sightings Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                sighting_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                confidence REAL NOT NULL,
                embedding_blob BLOB,
                keyframe_id TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """)

            # Relocalization Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS relocalizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                success INTEGER NOT NULL,
                matched_keyframe_id TEXT,
                inliers INTEGER,
                drift REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """)

            # Anomalies Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                type TEXT NOT NULL,
                description TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """)
            
            conn.commit()

    def start_session(self, session_id: str, start_time: float) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time) VALUES (?, ?);",
                (session_id, start_time)
            )
            conn.commit()

    def end_session(self, session_id: str, end_time: float, summary: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET end_time = ?, summary = ? WHERE session_id = ?;",
                (end_time, summary, session_id)
            )
            conn.commit()

    def log_pose(self, session_id: str, timestamp: float, pose: Tuple[float, float, float, float]) -> None:
        x, y, z, yaw = pose
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO trajectories (session_id, timestamp, x, y, z, yaw) VALUES (?, ?, ?, ?, ?, ?);",
                (session_id, timestamp, x, y, z, yaw)
            )
            conn.commit()

    def log_sighting(self, 
                     sighting_id: str, 
                     session_id: str, 
                     timestamp: float, 
                     label: str, 
                     position: Tuple[float, float, float], 
                     confidence: float, 
                     embedding: bytes, 
                     keyframe_id: str) -> None:
        x, y, z = position
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO sightings 
                (sighting_id, session_id, timestamp, label, x, y, z, confidence, embedding_blob, keyframe_id) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (sighting_id, session_id, timestamp, label, x, y, z, confidence, embedding, keyframe_id)
            )
            conn.commit()

    def log_relocalization(self, 
                           session_id: str, 
                           timestamp: float, 
                           success: bool, 
                           matched_keyframe_id: str, 
                           inliers: int, 
                           drift: float) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO relocalizations 
                (session_id, timestamp, success, matched_keyframe_id, inliers, drift) 
                VALUES (?, ?, ?, ?, ?, ?);""",
                (session_id, timestamp, 1 if success else 0, matched_keyframe_id, inliers, drift)
            )
            conn.commit()

    def log_anomaly(self, session_id: str, timestamp: float, anomaly_type: str, description: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO anomalies (session_id, timestamp, type, description) VALUES (?, ?, ?, ?);",
                (session_id, timestamp, anomaly_type, description)
            )
            conn.commit()

    def get_session_trajectory(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT timestamp, x, y, z, yaw FROM trajectories WHERE session_id = ? ORDER BY timestamp ASC;",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session_sightings(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT sighting_id, timestamp, label, x, y, z, confidence, keyframe_id FROM sightings WHERE session_id = ?;",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
