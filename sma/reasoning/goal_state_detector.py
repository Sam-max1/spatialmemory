from typing import Dict, Any, List

class GoalStateDetector:
    def __init__(self):
        pass

    def evaluate_goal_state(self, 
                            intent: str, 
                            retrieved_matches: List[Dict[str, Any]], 
                            searched_coverage: float) -> Dict[str, Any]:
        """
        Check if the retrieval and reasoning output meets completion requirements.
        Returns: {"goal_achieved": bool, "status": str}
        """
        if not retrieved_matches:
            if searched_coverage >= 0.95:
                return {
                    "goal_achieved": True,
                    "status": "definitive-not-found"
                }
            else:
                return {
                    "goal_achieved": False,
                    "status": "insufficient-coverage"
                }
                
        # If matches are found, check if they meet confidence requirements
        best_confidence = max([m.get("confidence", 0.0) for m in retrieved_matches])
        if best_confidence >= 0.7:
            return {
                "goal_achieved": True,
                "status": "target-located"
            }
            
        return {
            "goal_achieved": False,
            "status": "low-confidence"
        }
