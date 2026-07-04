from typing import List, Dict, Any

class LearningExtractor:
    def __init__(self):
        # Default permanence priors (half-life in days)
        self.class_permanence_priors = {
            "fire extinguisher": 365.0, # highly permanent
            "desk": 180.0,
            "cabinet": 120.0,
            "laptop": 2.0,            # moves frequently
            "chair": 14.0,
            "cup": 0.5                # very transient
        }

    def learn_permanence(self, sighting_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Adjust class permanence values based on observed coordinate movements.
        """
        # Simulated updating of priors
        learned_priors = self.class_permanence_priors.copy()
        
        # In mock mode, we slightly decay laptop and cup priors if movement is detected
        for sighting in sighting_history:
            label = sighting.get("label", "").lower()
            if label in learned_priors:
                # Mock update based on sightings counts
                learned_priors[label] = float(learned_priors[label] * 0.95)
                
        return learned_priors

    def identify_unreliable_zones(self, relocalizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify coordinates where Place Recognition consistently fails (e.g. glass partitions).
        """
        unreliable_zones = []
        
        # Analyze relocalization attempts
        failures = [r for r in relocalizations if not r.get("success", True)]
        for f in failures:
            # Group close failure points and label them
            unreliable_zones.append({
                "center": (f.get("x", 0.0), f.get("y", 0.0), f.get("z", 0.0)),
                "reason": "VPR matching failure - reflective surface suspect",
                "radius": 1.5
            })
            
        return unreliable_zones
