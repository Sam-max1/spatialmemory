from typing import List, Dict, Any, Tuple
import numpy as np

class Retriever:
    def __init__(self, ltm: Any, episodic: Any):
        self.ltm = ltm
        self.episodic = episodic

    def retrieve_places(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query place recognition index (VPR embeddings) to retrieve matched historical keyframes.
        """
        # In this mock implementation, we return registered nodes that match semantic VPR criteria.
        all_nodes = self.ltm.get_all_nodes()
        places = [n for n in all_nodes if n["type"] == "place"]
        
        matches = []
        for p in places:
            # Mock VPR cosine similarity score
            score = 0.85  # Default matched score
            matches.append({
                "node_id": p["node_id"],
                "label": p["label"],
                "position": (p["x"], p["y"], p["z"]),
                "confidence": p["confidence"],
                "similarity": score
            })
            
        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def retrieve_objects(self, query_label: str) -> List[Dict[str, Any]]:
        """
        Search long term memory and episodic checkpoints for specific objects.
        """
        all_nodes = self.ltm.get_all_nodes()
        results = []
        
        for n in all_nodes:
            if n["type"] == "object" and query_label.lower() in n["label"].lower():
                results.append({
                    "node_id": n["node_id"],
                    "label": n["label"],
                    "position": (n["x"], n["y"], n["z"]),
                    "confidence": n["confidence"],
                    "last_seen": n["last_seen"],
                    "metadata": n["metadata"]
                })
        return results

    def query_range(self, position: Tuple[float, float, float], radius: float) -> List[Dict[str, Any]]:
        """
        Perform spatial range query over long-term scene structures.
        """
        all_nodes = self.ltm.get_all_nodes()
        results = []
        qp = np.array(position)
        
        for n in all_nodes:
            if n["x"] is None or n["y"] is None or n["z"] is None:
                continue
            np_pos = np.array([n["x"], n["y"], n["z"]])
            dist = np.linalg.norm(qp - np_pos)
            if dist <= radius:
                results.append({
                    "node_id": n["node_id"],
                    "label": n["label"],
                    "type": n["type"],
                    "position": (n["x"], n["y"], n["z"]),
                    "distance": float(dist)
                })
        return results
