from typing import Dict, List, Any
from sma.retrieval.retriever import Retriever
from sma.retrieval.query_expander import QueryExpander

class RetrievalRouter:
    def __init__(self, retriever: Retriever, expander: QueryExpander):
        self.retriever = retriever
        self.expander = expander

    def route_query(self, intent: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Routes the retrieval pipeline based on the classification intent:
        - locate-object: expands synonyms, queries object index, and ranks results.
        - relocalize: generates rotation-probed VPR matches.
        - describe-region: performs range queries to capture neighboring objects.
        """
        if intent == "locate-object":
            label = params.get("label", "")
            expanded_labels = self.expander.expand_object_query(label)
            
            all_results = []
            seen_ids = set()
            for exp_label in expanded_labels:
                matches = self.retriever.retrieve_objects(exp_label)
                for m in matches:
                    if m["node_id"] not in seen_ids:
                        seen_ids.add(m["node_id"])
                        all_results.append(m)
            return all_results
            
        elif intent == "relocalize":
            # Uses place search with cosine distance
            embedding = params.get("embedding", None)
            if embedding is not None:
                return self.retriever.retrieve_places(embedding)
            return []
            
        elif intent == "describe-region":
            position = params.get("position", (0.0, 0.0, 0.0))
            radius = params.get("radius", 3.0)
            return self.retriever.query_range(position, radius)
            
        return []
