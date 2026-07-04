import numpy as np
from typing import List, Dict, Tuple, Any

class ObjectNode:
    def __init__(self, obj_id: str, label: str, position: Tuple[float, float, float], size: Tuple[float, float, float], confidence: float, embedding: np.ndarray):
        self.obj_id = obj_id
        self.label = label
        self.position = position  # (x, y, z)
        self.size = size          # (dx, dy, dz)
        self.confidence = confidence
        self.embedding = embedding
        self.metadata = {}

class GraphBuilder:
    def __init__(self, proximity_threshold: float = 2.0):
        self.proximity_threshold = proximity_threshold

    def extract_relations(self, objects: List[ObjectNode]) -> List[Tuple[str, str, str]]:
        """
        Extract physical relations between 3D-lifted object bounding boxes based on geometric distances.
        Relations returned: (subject_id, predicate, object_id)
        Predicates: 'on', 'in', 'near', 'attached_to'
        """
        relations = []
        n = len(objects)
        for i in range(n):
            obj_a = objects[i]
            pos_a = np.array(obj_a.position)
            sz_a = np.array(obj_a.size)
            
            for j in range(n):
                if i == j:
                    continue
                obj_b = objects[j]
                pos_b = np.array(obj_b.position)
                sz_b = np.array(obj_b.size)
                
                # Vector from B to A
                delta = pos_a - pos_b
                dist = np.linalg.norm(delta)
                
                # Check height relationship (Z is vertical axis)
                z_dist = delta[2] # A's z minus B's z
                horizontal_dist = np.linalg.norm(delta[:2])
                
                # Check if A is vertically directly above B and close (ON)
                is_above = z_dist > 0 and z_dist < (sz_a[2]/2 + sz_b[2]/2 + 0.15)
                is_aligned_horizontally = horizontal_dist < max(sz_a[0], sz_b[0], 0.5)
                
                if is_above and is_aligned_horizontally:
                    # Predicate: A is "ON" B (e.g. laptop is ON desk)
                    relations.append((obj_a.obj_id, "on", obj_b.obj_id))
                elif horizontal_dist < (sz_a[0]/2 + sz_b[0]/2 + 0.2) and abs(z_dist) < max(sz_a[2], sz_b[2]):
                    # Predicate: A is "NEAR" B
                    relations.append((obj_a.obj_id, "near", obj_b.obj_id))
                    
                # Support "IN" (e.g., container relations)
                # If B is large and contains A's centroid
                if obj_b.label in ["cabinet", "drawer", "box", "refrigerator", "room"]:
                    half_sz_b = sz_b / 2.0
                    contains_x = abs(delta[0]) < half_sz_b[0]
                    contains_y = abs(delta[1]) < half_sz_b[1]
                    contains_z = abs(delta[2]) < half_sz_b[2]
                    if contains_x and contains_y and contains_z:
                        relations.append((obj_a.obj_id, "in", obj_b.obj_id))

        return relations
