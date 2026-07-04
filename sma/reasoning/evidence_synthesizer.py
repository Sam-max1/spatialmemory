import numpy as np
from typing import List, Dict, Any, Tuple

class EvidenceSynthesizer:
    def __init__(self, movement_threshold: float = 0.5):
        self.movement_threshold = movement_threshold

    def resolve_displacement(self, 
                             sighting_a: Dict[str, Any], 
                             sighting_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if two sightings represent:
        - "moved": The same object moved to a new coordinate.
        - "instances": Two distinct copies of the object exist simultaneously.
        Decision rules:
        - If coordinate distance < movement_threshold: same object (slight tracking drift).
        - If embedding similarity is very high but distance is large, check timestamp:
          if timestamp difference is large, it "moved"; if timestamps are overlaps/near, they are "instances".
        """
        pos_a = np.array(sighting_a["position"])
        pos_b = np.array(sighting_b["position"])
        dist = np.linalg.norm(pos_a - pos_b)

        if dist < self.movement_threshold:
            # Same object, average position
            avg_pos = (pos_a + pos_b) / 2.0
            return {
                "relation": "identity",
                "resolved_position": avg_pos.tolist(),
                "confidence": min(sighting_a["confidence"], sighting_b["confidence"])
            }

        # Check timestamp ordering
        t_a = sighting_a.get("timestamp", 0)
        t_b = sighting_b.get("timestamp", 0)
        
        # If they were seen at the same time, they are two instances
        if abs(t_a - t_b) < 10.0:  # within 10 seconds (same session path)
            return {
                "relation": "instances",
                "resolved_positions": [sighting_a["position"], sighting_b["position"]],
                "confidence": min(sighting_a["confidence"], sighting_b["confidence"])
            }
        
        # If seen at different times, assume it moved
        newer = sighting_b if t_b > t_a else sighting_a
        older = sighting_a if t_b > t_a else sighting_b
        return {
            "relation": "moved",
            "resolved_position": newer["position"],
            "previous_position": older["position"],
            "confidence": newer["confidence"],
            "last_seen": newer.get("timestamp")
        }

    def fuse_voxels(self, 
                    voxel_a: Dict[str, Any], 
                    voxel_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Confidence-weighted TSDF voxel fusion with recency bias.
        """
        tsdf_a, conf_a, t_a = voxel_a["tsdf"], voxel_a["confidence"], voxel_a["timestamp"]
        tsdf_b, conf_b, t_b = voxel_b["tsdf"], voxel_b["confidence"], voxel_b["timestamp"]

        # Recency scaling: weight decays by 20% per hour of age difference
        time_diff_hours = abs(t_a - t_b) / 3600.0
        recency_weight = 1.0 / (1.0 + 0.2 * time_diff_hours)

        if t_a > t_b:
            # Voxel A is newer
            weight_a = conf_a
            weight_b = conf_b * recency_weight
        else:
            # Voxel B is newer
            weight_a = conf_a * recency_weight
            weight_b = conf_b

        total_weight = weight_a + weight_b
        if total_weight == 0:
            return {"tsdf": 0.0, "confidence": 0.0, "timestamp": max(t_a, t_b)}

        fused_tsdf = (tsdf_a * weight_a + tsdf_b * weight_b) / total_weight
        fused_conf = min(1.0, total_weight)

        return {
            "tsdf": float(fused_tsdf),
            "confidence": float(fused_conf),
            "timestamp": max(t_a, t_b)
        }
