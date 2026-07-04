from typing import List, Dict, Any

class QueryExpander:
    def __init__(self):
        # A simple local synonym mapping for open-vocabulary expansion
        self.synonym_map = {
            "fire extinguisher": ["extinguisher", "red canister", "safety bottle"],
            "laptop": ["computer", "notebook", "pc", "macbook"],
            "chair": ["seat", "stool", "armchair"],
            "desk": ["table", "workbench", "workstation"],
            "refrigerator": ["fridge", "freezer", "cooler"]
        }

    def expand_object_query(self, label: str) -> List[str]:
        """
        Expand object search terms with synonyms to increase retrieval recall.
        """
        terms = [label.lower()]
        for key, synonyms in self.synonym_map.items():
            if label.lower() == key or label.lower() in synonyms:
                if key not in terms:
                    terms.append(key)
                for s in synonyms:
                    if s not in terms:
                        terms.append(s)
        return terms

    def generate_rotation_variants(self, pose_yaw: float, steps: int = 4) -> List[float]:
        """
        Generate rotation variants (in radians) to handle place recognition searches
        when approaching a location from different directions.
        """
        import math
        variants = []
        for i in range(steps):
            angle = pose_yaw + (i * (2 * math.pi / steps))
            # Normalize to [-pi, pi]
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            variants.append(angle)
        return variants
