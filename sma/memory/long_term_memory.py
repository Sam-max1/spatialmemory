import sqlite3
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

class LongTermMemory:
    def __init__(self, scene_db_path: Path, voxel_db_path: Path):
        self.scene_db_path = scene_db_path
        self.voxel_db_path = voxel_db_path
        self._init_scene_db()
        self._init_voxel_db()

    def _get_scene_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.scene_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_scene_db(self) -> None:
        with self._get_scene_connection() as conn:
            # Nodes Table (Site, Floor, Room, Surface, Object)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                x REAL,
                y REAL,
                z REAL,
                confidence REAL,
                last_seen REAL,
                embedding_blob BLOB,
                metadata TEXT
            );
            """)

            # Edges Table (Relations)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT,
                target_id TEXT,
                relation TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, relation),
                FOREIGN KEY(source_id) REFERENCES nodes(node_id),
                FOREIGN KEY(target_id) REFERENCES nodes(node_id)
            );
            """)
            conn.commit()

    def _init_voxel_db(self) -> None:
        # Since RocksDB can have complex compilation requirements,
        # we implement a persistent dictionary-based voxel database backed by pickle.
        self.voxel_cache: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        if self.voxel_db_path.exists():
            try:
                with open(self.voxel_db_path, "rb") as f:
                    self.voxel_cache = pickle.load(f)
            except Exception:
                self.voxel_cache = {}

    def save_voxels(self) -> None:
        with open(self.voxel_db_path, "wb") as f:
            pickle.dump(self.voxel_cache, f)

    # Node operations
    def upsert_node(self, 
                    node_id: str, 
                    node_type: str, 
                    label: str, 
                    position: Tuple[float, float, float], 
                    confidence: float, 
                    last_seen: float, 
                    embedding: Optional[bytes] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        x, y, z = position
        meta_str = json.dumps(metadata or {})
        with self._get_scene_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO nodes 
                (node_id, type, label, x, y, z, confidence, last_seen, embedding_blob, metadata) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (node_id, node_type, label, x, y, z, confidence, last_seen, embedding, meta_str)
            )
            conn.commit()

    # Edge operations
    def add_edge(self, source_id: str, target_id: str, relation: str) -> None:
        with self._get_scene_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, relation) VALUES (?, ?, ?);",
                (source_id, target_id, relation)
            )
            conn.commit()

    def remove_edge(self, source_id: str, target_id: str, relation: str) -> None:
        with self._get_scene_connection() as conn:
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND relation = ?;",
                (source_id, target_id, relation)
            )
            conn.commit()

    # Voxel operations
    def set_voxel(self, coords: Tuple[int, int, int], tsdf: float, confidence: float, timestamp: float) -> None:
        self.voxel_cache[coords] = {
            "tsdf": tsdf,
            "confidence": confidence,
            "timestamp": timestamp
        }

    def get_voxel(self, coords: Tuple[int, int, int]) -> Optional[Dict[str, Any]]:
        return self.voxel_cache.get(coords)

    # Retrieval operations
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        with self._get_scene_connection() as conn:
            cursor = conn.execute("SELECT * FROM nodes;")
            nodes = []
            for r in cursor.fetchall():
                node = dict(r)
                node["metadata"] = json.loads(node["metadata"])
                nodes.append(node)
            return nodes

    def get_all_edges(self) -> List[Tuple[str, str, str]]:
        with self._get_scene_connection() as conn:
            cursor = conn.execute("SELECT source_id, target_id, relation FROM edges;")
            return [(r["source_id"], r["target_id"], r["relation"]) for r in cursor.fetchall()]
