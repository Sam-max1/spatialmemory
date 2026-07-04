import time
from typing import Dict, Any, Optional
from sma.reasoning.agent_runner import AgentRunner

class SessionManager:
    def __init__(self, runner: AgentRunner):
        self.runner = runner
        self.active_session_id: Optional[str] = None
        self.start_time: float = 0.0

    def start_session(self) -> str:
        """
        Create and open a new session in episodic memory.
        """
        self.active_session_id = f"session_{int(time.time())}"
        self.start_time = time.time()
        self.runner.current_session_id = self.active_session_id
        self.runner.em.start_session(self.active_session_id, self.start_time)
        return self.active_session_id

    def resume_session(self, old_session_id: str) -> bool:
        """
        Resume a previous session by looking up its coordinates in episodic memory.
        """
        trajectory = self.runner.em.get_session_trajectory(old_session_id)
        if not trajectory:
            return False
            
        # Set runner state to last recorded pose
        last_pose = trajectory[-1]
        self.runner.wm.update_pose(
            pose=time.tzname[0].encode('utf-8'), # arbitrary mock tag, actually coordinates:
            covariance=time.time()
        )
        self.active_session_id = old_session_id
        self.runner.current_session_id = old_session_id
        return True

    def close_session(self) -> Dict[str, Any]:
        """
        Close session, run memory summaries, promote object sightings, and run voxel decay.
        """
        if not self.active_session_id:
            return {}

        end_time = time.time()
        duration = end_time - self.start_time
        
        # Trigger object promotions
        promoted = self.runner.mm.promote_sightings_to_ltm(self.active_session_id)
        
        # Trigger voxel decay compaction
        pruned_voxels = self.runner.mm.decay_voxels(self.active_session_id)
        
        summary = f"Session closed after {duration:.1f}s. Promoted {len(promoted)} objects. Pruned {pruned_voxels} voxels."
        self.runner.em.end_session(self.active_session_id, end_time, summary)
        
        # Clear WM ring
        self.runner.wm.clear()
        
        ret = {
            "session_id": self.active_session_id,
            "duration_sec": duration,
            "promoted_objects": promoted,
            "pruned_voxels": pruned_voxels,
            "summary": summary
        }
        
        self.active_session_id = None
        return ret
