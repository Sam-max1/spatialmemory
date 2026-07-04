import numpy as np
from typing import Tuple, List, Dict, Any

class Chunker:
    def __init__(self, novelty_threshold: float = 0.25, info_gain_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold
        self.info_gain_threshold = info_gain_threshold
        self.last_keyframe_embedding = None

    def evaluate_keyframe(self, frame_embedding: np.ndarray, current_voxels: int, historical_voxels: int) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluate if a frame is a keyframe based on visual novelty and voxel information gain.
        """
        # Calculate novelty
        if self.last_keyframe_embedding is None:
            novelty = 1.0
        else:
            # Cosine distance
            dot_product = np.dot(frame_embedding, self.last_keyframe_embedding)
            norm_a = np.linalg.norm(frame_embedding)
            norm_b = np.linalg.norm(self.last_keyframe_embedding)
            if norm_a == 0 or norm_b == 0:
                novelty = 1.0
            else:
                novelty = 1.0 - (dot_product / (norm_a * norm_b))

        # Calculate info gain (fraction of new voxels relative to current map density)
        voxel_diff = current_voxels - historical_voxels
        info_gain = voxel_diff / max(1, historical_voxels)

        is_keyframe = bool((novelty > self.novelty_threshold) or (info_gain > self.info_gain_threshold))

        if is_keyframe:
            self.last_keyframe_embedding = frame_embedding

        metrics = {
            "novelty": float(novelty),
            "info_gain": float(info_gain)
        }

        return is_keyframe, metrics

    def partition_voxels(self, points_3d: np.ndarray, voxel_size: float = 0.05) -> Dict[Tuple[int, int, int], List[np.ndarray]]:
        """
        Partition 3D points into 8x8x8 voxel blocks.
        """
        partitioned = {}
        if points_3d.size == 0:
            return partitioned

        # Group points into coordinates
        indices = np.floor(points_3d / voxel_size).astype(int)
        for i, idx in enumerate(indices):
            block_coords = (idx[0] // 8, idx[1] // 8, idx[2] // 8)
            if block_coords not in partitioned:
                partitioned[block_coords] = []
            partitioned[block_coords].append(points_3d[i])

        return partitioned
