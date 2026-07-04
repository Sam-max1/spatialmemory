import re
from typing import Dict, Any

class ConstraintExtractor:
    def __init__(self):
        pass

    def extract_constraints(self, query_text: str) -> Dict[str, Any]:
        """
        Extract spatial and temporal bounds from natural language queries:
        - "on floor 2" -> {"floor": 2}
        - "since yesterday" -> {"time_horizon": "24h"}
        - "only if you're sure" -> {"min_confidence": 0.85}
        """
        txt = query_text.lower()
        constraints = {
            "floor": None,
            "time_horizon": None,
            "min_confidence": 0.5
        }

        # Match floor numbers
        floor_match = re.search(r"floor\s*(\d+)", txt)
        if floor_match:
            constraints["floor"] = int(floor_match.group(1))

        # Match recency conditions
        if "since yesterday" in txt or "within 24 hours" in txt:
            constraints["time_horizon"] = 24.0 * 3600.0  # seconds
        elif "within 1 hour" in txt or "just now" in txt:
            constraints["time_horizon"] = 3600.0

        # Match confidence assertions
        if "sure" in txt or "certain" in txt or "high confidence" in txt:
            constraints["min_confidence"] = 0.85
        elif "any" in txt or "rough" in txt:
            constraints["min_confidence"] = 0.10

        return constraints
