from typing import Dict, Any

class FeedbackIntegrator:
    def __init__(self, ltm: Any):
        self.ltm = ltm

    def integrate_operator_correction(self, node_id: str, corrective_label: str) -> bool:
        """
        Manually adjust object node descriptions based on operator correction commands.
        """
        all_nodes = self.ltm.get_all_nodes()
        target_node = next((n for n in all_nodes if n["node_id"] == node_id), None)
        
        if not target_node:
            return False
            
        # Update node with corrected label and set confidence to max
        self.ltm.upsert_node(
            node_id=node_id,
            node_type=target_node["type"],
            label=corrective_label,
            position=(target_node["x"], target_node["y"], target_node["z"]),
            confidence=1.0,
            last_seen=target_node["last_seen"],
            metadata={**target_node.get("metadata", {}), "operator_corrected": True}
        )
        return True
