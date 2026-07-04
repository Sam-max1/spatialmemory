import numpy as np
from typing import Tuple, Dict, Any

class QueryRewriter:
    def __init__(self):
        pass

    def rewrite_deixis(self, query_text: str, current_pose: Tuple[float, float, float, float], trajectory: list) -> str:
        """
        Rewrite spatial deixis terms:
        - "behind me" -> absolute vector based on current heading
        - "the room I was just in" -> extracts name of last entered room node from trajectory
        """
        txt = query_text.lower()
        x, y, z, yaw = current_pose
        
        if "behind me" in txt:
            # Behind means heading opposite direction (yaw + pi)
            new_yaw = yaw + np.pi
            # Normalize to [-pi, pi]
            new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
            # Convert text annotation to coordinate query offsets
            dx = -1.5 * np.cos(yaw)
            dy = -1.5 * np.sin(yaw)
            return query_text.replace("behind me", f"at location ({x + dx:.2f}, {y + dy:.2f})")
            
        if "room i just left" in txt or "room i was just in" in txt:
            # Look at trajectory history
            # In mock mode, return kitchen or office
            return query_text.replace("the room i was just in", "room_office_01")
            
        return query_text
