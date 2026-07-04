from typing import Dict, Any, List

class MemoryUpdater:
    def __init__(self, ltm: Any):
        self.ltm = ltm
        self.journal_log: List[Dict[str, Any]] = []

    def update_permanence_priors(self, priors: Dict[str, float]) -> None:
        """
        Write updated object permanence priors into the LTM metadata store.
        """
        # Save priors inside LTM site calibration/metadata nodes
        self.ltm.upsert_node(
            node_id="calibration_permanence_priors",
            node_type="calibration",
            label="permanence_priors",
            position=(0.0, 0.0, 0.0),
            confidence=1.0,
            last_seen=0.0,
            metadata=priors
        )
        self.journal_log.append({
            "operation": "update_priors",
            "priors_count": len(priors)
        })

    def revert_last_update(self) -> bool:
        """
        Revert the last transaction in the journal if drift metrics regress.
        """
        if not self.journal_log:
            return False
            
        last_op = self.journal_log.pop()
        # Mocking revert logic
        return True
