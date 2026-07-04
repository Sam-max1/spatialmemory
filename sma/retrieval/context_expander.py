from typing import Dict, List, Any, Optional

class ContextExpander:
    def __init__(self, ltm: Any):
        self.ltm = ltm

    def expand_object_context(self, object_node_id: str) -> Dict[str, Any]:
        """
        Traverse the scene graph to find an object's spatial context:
        its parent surface, parent room, connecting portals, and neighboring objects.
        """
        all_nodes = self.ltm.get_all_nodes()
        all_edges = self.ltm.get_all_edges()

        # Find target node
        target_node = None
        for n in all_nodes:
            if n["node_id"] == object_node_id:
                target_node = n
                break

        if not target_node:
            return {}

        context = {
            "node_id": object_node_id,
            "label": target_node["label"],
            "position": (target_node["x"], target_node["y"], target_node["z"]),
            "parent_surface": None,
            "parent_room": None,
            "neighbors": [],
            "connecting_portals": []
        }

        # Find parent surface or room by edges (e.g., source_id = target_node_id, relation = 'on' or 'in')
        # Relations: (subject, predicate, object). If target is ON surface: (target_node_id, 'on', surface_id)
        # If target is IN room: (target_node_id, 'in', room_id)
        for src, dest, rel in all_edges:
            if src == object_node_id:
                # Find dest node details
                dest_node = next((n for n in all_nodes if n["node_id"] == dest), None)
                if dest_node:
                    if rel == "on":
                        context["parent_surface"] = {
                            "node_id": dest,
                            "label": dest_node["label"],
                            "position": (dest_node["x"], dest_node["y"], dest_node["z"])
                        }
                    elif rel == "in":
                        context["parent_room"] = {
                            "node_id": dest,
                            "label": dest_node["label"]
                        }
            
            # Find neighbors (nodes sharing the same parent room or surface)
            # Or connected via a 'near' edge
            if (src == object_node_id or dest == object_node_id) and rel == "near":
                neigh_id = dest if src == object_node_id else src
                neigh_node = next((n for n in all_nodes if n["node_id"] == neigh_id), None)
                if neigh_node:
                    context["neighbors"].append({
                        "node_id": neigh_id,
                        "label": neigh_node["label"],
                        "position": (neigh_node["x"], neigh_node["y"], neigh_node["z"])
                    })

        return context
