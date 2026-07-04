from typing import List, Dict, Any, Tuple
import numpy as np

class Reranker:
    def __init__(self, min_inliers: int = 30, max_error_px: float = 2.0):
        self.min_inliers = min_inliers
        self.max_error_px = max_error_px

    def verify_geometry(self, query_features: np.ndarray, candidate_features: np.ndarray) -> Tuple[bool, int, float]:
        """
        Verify spatial consensus between frame keypoints using a simulated PnP / RANSAC algorithm.
        Returns: (success, inlier_count, residual_error)
        """
        # In a real environment, this runs cv2.solvePnPRansac or feature matching.
        # We simulate the geometric consistency matching.
        # Let's compute a deterministic match count based on feature differences.
        if query_features.shape != candidate_features.shape:
            return False, 0, 999.0
            
        diff = np.linalg.norm(query_features - candidate_features)
        
        # Lower difference = more consistent keypoint alignment
        simulated_inliers = int(max(0, 100 - (diff * 50)))
        simulated_residual = float(diff * 0.1)
        
        success = (simulated_inliers >= self.min_inliers) and (simulated_residual < self.max_error_px)
        
        return success, simulated_inliers, simulated_residual

    def rank_sightings(self, sightings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank candidate sightings based on joint recency and visual detection confidence.
        Formula: score = confidence * (1.0 / (1.0 + time_delta))
        """
        import time
        now = time.time()
        
        ranked = []
        for s in sightings:
            last_seen = s.get("last_seen", now)
            time_delta = max(0.0, now - last_seen)
            # Normalize time decay: half-life of 3600 seconds
            time_decay = 1.0 / (1.0 + (time_delta / 3600.0))
            
            score = s["confidence"] * time_decay
            
            s_copy = s.copy()
            s_copy["ranking_score"] = float(score)
            ranked.append(s_copy)
            
        ranked.sort(key=lambda x: x["ranking_score"], reverse=True)
        return ranked
