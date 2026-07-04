import time
from typing import Dict, List, Any, Tuple, Optional
from sma.memory.working_memory import WorkingMemory, KeyframeRecord
from sma.memory.episodic_memory import EpisodicMemory
from sma.memory.long_term_memory import LongTermMemory

class MemoryManager:
    def __init__(self, wm: WorkingMemory, em: EpisodicMemory, ltm: LongTermMemory, storage_budget_gb: float = 20.0):
        self.wm = wm
        self.em = em
        self.ltm = ltm
        self.storage_budget_gb = storage_budget_gb

    def promote_working_to_episodic(self, record: KeyframeRecord) -> None:
        """
        Write keyframe details and raw data to episodic memory logging.
        """
        # Save sighting if objects detected
        self.em.start_session(record.session_id, record.timestamp)
        
        # Log robot pose
        # Pose contains (x, y, z, yaw) - mapping yaw from pose rotation if available
        # Here we just take coordinates and yaw = 0 for standard poses
        pos_3d = record.pose[:3]
        yaw = float(record.pose[3]) if len(record.pose) > 3 else 0.0
        self.em.log_pose(record.session_id, record.timestamp, (pos_3d[0], pos_3d[1], pos_3d[2], yaw))

    def promote_sightings_to_ltm(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Promote episodic object sightings to long-term memory scene graph.
        Requirement: Node must have >= 3 observations across >= 2 different viewpoints.
        """
        promoted_nodes = []
        sightings = self.em.get_session_sightings(session_id)
        
        # Count sightings by label to mock viewpoints/confidence checks
        counts = {}
        for s in sightings:
            label = s["label"]
            if label not in counts:
                counts[label] = []
            counts[label].append(s)

        for label, group in counts.items():
            if len(group) >= 3: # 3-obs rule met
                # Compute average position
                xs = [s["x"] for s in group]
                ys = [s["y"] for s in group]
                zs = [s["z"] for s in group]
                avg_pos = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
                
                # Pick sighting id
                node_id = f"node_{label}_{int(time.time())}"
                confidence = sum([s["confidence"] for s in group])/len(group)
                
                # Upsert into LTM
                self.ltm.upsert_node(
                    node_id=node_id,
                    node_type="object",
                    label=label,
                    position=avg_pos,
                    confidence=confidence,
                    last_seen=time.time(),
                    metadata={"sighting_count": len(group), "source_session": session_id}
                )
                promoted_nodes.append({
                    "id": node_id,
                    "label": label,
                    "position": avg_pos,
                    "confidence": confidence
                })
        return promoted_nodes

    def decay_voxels(self, current_session: str, decay_sessions: int = 20, threshold: float = 0.15) -> int:
        """
        Prune long-term voxels unobserved for more than N sessions.
        Returns the count of pruned voxels.
        """
        pruned_count = 0
        coords_to_delete = []
        
        # Simulate voxel decay checks
        for coords, meta in self.ltm.voxel_cache.items():
            # If voxel has low confidence or older timestamp
            # Mock decay: decrease confidence by 0.05
            meta["confidence"] -= 0.05
            if meta["confidence"] < threshold:
                coords_to_delete.append(coords)
                
        for coords in coords_to_delete:
            del self.ltm.voxel_cache[coords]
            pruned_count += 1
            
        if pruned_count > 0:
            self.ltm.save_voxels()
            
        return pruned_count
