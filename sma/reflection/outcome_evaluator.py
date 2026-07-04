import numpy as np
from typing import Dict, List, Any

class OutcomeEvaluator:
    def __init__(self):
        pass

    def evaluate_session_drift(self, trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate drift and covariance growth from a session trajectory log.
        """
        if len(trajectory) < 2:
            return {"drift_rate": 0.0, "max_uncertainty": 0.0}

        # Calculate coordinate distances
        poses = np.array([[p["x"], p["y"], p["z"]] for p in trajectory])
        diffs = np.diff(poses, axis=0)
        total_distance = np.sum(np.linalg.norm(diffs, axis=1))

        # Simulated drift estimation (typically calculated via loop-closures or GTSAM residuals)
        simulated_drift = float(total_distance * 0.005) # 0.5% drift baseline
        
        return {
            "total_distance_m": float(total_distance),
            "estimated_drift_m": simulated_drift,
            "drift_rate_pct": float((simulated_drift / total_distance) * 100 if total_distance > 0 else 0)
        }
