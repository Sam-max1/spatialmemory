import numpy as np
from typing import List, Tuple

class ContentFilter:
    def __init__(self):
        pass

    def verify_waypoints(self, waypoints: List[Tuple[int, int]], occupancy_grid: np.ndarray) -> bool:
        """
        Verify that a series of navigation waypoints does not intersect with known obstacles
        in the metric occupancy grid (Safety critical output check).
        """
        rows, cols = occupancy_grid.shape
        for wp in waypoints:
            r, c = wp
            if not (0 <= r < rows and 0 <= c < cols):
                return False  # out of bounds
            if occupancy_grid[r, c] == 1:
                return False  # hits obstacle
        return True
