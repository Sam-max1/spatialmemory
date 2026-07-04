import heapq
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ToolCaller:
    def __init__(self):
        pass

    def plan_path(self, 
                  occupancy_grid: np.ndarray, 
                  start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Execute A* path planning on a 2D occupancy grid.
        occupancy_grid: numpy array where 0 is empty, 1 is obstacle.
        start, goal: coordinates (row, col) as grid indices.
        Returns: (path, success)
        """
        rows, cols = occupancy_grid.shape
        if not (0 <= start[0] < rows and 0 <= start[1] < cols):
            return [], False
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            return [], False
        if occupancy_grid[start] == 1 or occupancy_grid[goal] == 1:
            return [], False

        # Open set as a priority queue: (f_score, (row, col))
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Maps nodes to their parent nodes
        came_from = {}
        
        # g_score: cost of cheapest path from start to node
        g_score = {start: 0}
        
        # f_score: current estimate of cost of cheapest path from start to goal through node
        # Heuristic: Manhattan distance
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        f_score = {start: h(start)}

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, True

            closed_set.add(current)
            
            # Neighbors (up, down, left, right, diagonals)
            neighbors = [
                (current[0]+1, current[1]), (current[0]-1, current[1]),
                (current[0], current[1]+1), (current[0], current[1]-1),
                (current[0]+1, current[1]+1), (current[0]-1, current[1]-1),
                (current[0]+1, current[1]-1), (current[0]-1, current[1]+1)
            ]
            
            for n in neighbors:
                if not (0 <= n[0] < rows and 0 <= n[1] < cols):
                    continue
                if occupancy_grid[n] == 1:
                    continue
                if n in closed_set:
                    continue
                
                # Diagonal moves cost sqrt(2) (~1.41), cardinal moves cost 1
                is_diagonal = (n[0] != current[0]) and (n[1] != current[1])
                move_cost = 1.41 if is_diagonal else 1.0
                tentative_g_score = g_score[current] + move_cost
                
                if tentative_g_score < g_score.get(n, float('inf')):
                    came_from[n] = current
                    g_score[n] = tentative_g_score
                    f_score[n] = tentative_g_score + h(n)
                    
                    # Add to open set if not present
                    if n not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[n], n))

        return [], False

    def export_map(self, voxels: Dict[Tuple[int, int, int], Dict[str, Any]], export_format: str = "ply") -> str:
        """
        Export long term voxels as PLY / GLB files.
        """
        if not voxels:
            return "# PLY empty map"
            
        ply_header = """ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        vertices = []
        for coords, meta in voxels.items():
            # Apply voxel coordinate scaling (e.g. 5cm resolution)
            x = coords[0] * 0.05
            y = coords[1] * 0.05
            z = coords[2] * 0.05
            # TSDF weight maps to red/green/blue
            tsdf = meta["tsdf"]
            r = int(max(0, min(255, (tsdf + 1.0) * 127.5)))
            g = int(max(0, min(255, meta["confidence"] * 255)))
            b = 150
            vertices.append(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}")

        formatted_ply = ply_header.format(vertex_count=len(vertices)) + "\n".join(vertices)
        return formatted_ply
