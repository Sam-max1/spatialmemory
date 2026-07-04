from typing import Dict, List, Any, Optional

class HierarchyMapper:
    def __init__(self):
        pass

    def map_hierarchy(self, 
                     site_id: str, 
                     objects: List[Dict[str, Any]], 
                     rooms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organize a flat list of objects and room segments into a nested spatial taxonomy:
        Site -> Floor -> Room -> Surface -> Object
        """
        hierarchy = {
            "id": site_id,
            "type": "site",
            "floors": {}
        }
        
        # In this simple model, we assume objects are mapped to rooms based on geographical inclusion.
        # Room schema: {"id": "room_kitchen", "floor": 1, "bounds": [x_min, y_min, x_max, y_max]}
        # Object schema: {"id": "obj_01", "label": "toaster", "position": [x, y, z]}
        
        for room in rooms:
            floor_num = room.get("floor", 1)
            if floor_num not in hierarchy["floors"]:
                hierarchy["floors"][floor_num] = {
                    "id": f"floor_{floor_num}",
                    "type": "floor",
                    "rooms": {}
                }
            
            hierarchy["floors"][floor_num]["rooms"][room["id"]] = {
                "id": room["id"],
                "name": room.get("name", room["id"]),
                "type": "room",
                "surfaces": {},
                "objects": []
            }

        # Place objects in matching rooms (defaulting to a generic room if outside bounds)
        for obj in objects:
            pos = obj.get("position", [0.0, 0.0, 0.0])
            placed = False
            
            for room in rooms:
                bounds = room.get("bounds", [-10, -10, 10, 10])
                if bounds[0] <= pos[0] <= bounds[2] and bounds[1] <= pos[1] <= bounds[3]:
                    floor_num = room.get("floor", 1)
                    r_id = room["id"]
                    
                    # Surfaces can be chairs, desks, counters
                    if obj.get("label") in ["desk", "table", "counter", "shelf"]:
                        surface_node = {
                            "id": obj["id"],
                            "label": obj["label"],
                            "type": "surface",
                            "objects": []
                        }
                        hierarchy["floors"][floor_num]["rooms"][r_id]["surfaces"][obj["id"]] = surface_node
                    else:
                        hierarchy["floors"][floor_num]["rooms"][r_id]["objects"].append(obj)
                    placed = True
                    break
                    
            if not placed:
                # Place in a fallback room on Floor 1
                fallback_floor = 1
                if fallback_floor not in hierarchy["floors"]:
                    hierarchy["floors"][fallback_floor] = {
                        "id": f"floor_{fallback_floor}",
                        "type": "floor",
                        "rooms": {
                            "room_hallway": {
                                "id": "room_hallway",
                                "name": "Main Hallway",
                                "type": "room",
                                "surfaces": {},
                                "objects": []
                            }
                        }
                    }
                r_id = list(hierarchy["floors"][fallback_floor]["rooms"].keys())[0]
                hierarchy["floors"][fallback_floor]["rooms"][r_id]["objects"].append(obj)

        return hierarchy
