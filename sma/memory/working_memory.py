import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class KeyframeRecord:
    frame_id: str
    timestamp: float
    image_data: bytes
    embedding: np.ndarray
    pose: np.ndarray
    pose_covariance: np.ndarray
    scale_confidence: str
    privacy_flags: Dict[str, bool]
    novelty_score: float
    info_gain: float

@dataclass
class WorkingMemoryState:
    session_id: str = "default_session"
    current_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pose_covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    keyframe_ring: List[KeyframeRecord] = field(default_factory=list)
    tracklets: Dict[str, Any] = field(default_factory=dict)
    active_query_context: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

class WorkingMemory:
    def __init__(self, max_keyframes: int = 64):
        self.max_keyframes = max_keyframes
        self.state = WorkingMemoryState()

    def update_pose(self, pose: np.ndarray, covariance: np.ndarray) -> None:
        self.state.current_pose = pose
        self.state.pose_covariance = covariance
        self.state.last_updated = time.time()

    def add_keyframe(self, record: KeyframeRecord) -> Optional[KeyframeRecord]:
        """
        Add a keyframe to the ring buffer.
        Returns the evicted keyframe if the buffer was full.
        """
        evicted = None
        if len(self.state.keyframe_ring) >= self.max_keyframes:
            # FIFO eviction
            evicted = self.state.keyframe_ring.pop(0)
        
        self.state.keyframe_ring.append(record)
        return evicted

    def update_tracklet(self, track_id: str, label: str, position: np.ndarray, confidence: float) -> None:
        self.state.tracklets[track_id] = {
            "label": label,
            "position": position.tolist(),
            "confidence": confidence,
            "last_seen": time.time()
        }

    def clear(self) -> None:
        self.state = WorkingMemoryState()
