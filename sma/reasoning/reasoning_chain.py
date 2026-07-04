from typing import List, Dict, Any

class ReasoningChain:
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode

    def process_cot(self, 
                    query: str, 
                    intent: str, 
                    retrieved_data: List[Dict[str, Any]], 
                    constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs Spatial Chain-of-Thought (CoT) over retrieved scene graph coordinates.
        Enforces rule: Block locations from parametric LLM knowledge without a valid provenance chain.
        """
        steps = []
        answer = ""
        provenance_nodes = []
        confidence = 0.0

        steps.append(f"Step 1: Parsing query '{query}' for intent '{intent}'.")
        steps.append(f"Step 2: Checking constraint filters: {constraints}.")

        if not retrieved_data:
            steps.append("Step 3: No records found in place recognition or object indexes.")
            steps.append("Step 4: Rejecting parametric assumptions to avoid hallucinations.")
            answer = "I could not locate that object. Based on my memory index, no observations match this description."
            return {
                "answer": answer,
                "chain_of_thought": steps,
                "provenance": [],
                "confidence": 0.0
            }

        # Filter by constraints (e.g. minimum confidence)
        filtered = [d for d in retrieved_data if d.get("confidence", 1.0) >= constraints.get("min_confidence", 0.5)]
        steps.append(f"Step 3: Filtered matches by confidence threshold. Checked {len(retrieved_data)} items, {len(filtered)} remaining.")

        if not filtered:
            steps.append("Step 4: All matching objects failed confidence/re-ranking thresholds.")
            answer = "I found matching records, but they fell below the confidence safety thresholds."
            return {
                "answer": answer,
                "chain_of_thought": steps,
                "provenance": [],
                "confidence": 0.0
            }

        # Select best node
        best_match = max(filtered, key=lambda x: x.get("confidence", 0.0))
        steps.append(f"Step 4: Selected highest confidence match node '{best_match['node_id']}' with confidence {best_match.get('confidence'):.2f}.")

        # Generate response using graph provenance details
        pos = best_match["position"]
        label = best_match["label"]
        provenance_nodes.append(best_match["node_id"])
        confidence = best_match["confidence"]

        steps.append("Step 5: Fusing graph and coordinate data. Formulating response.")
        
        # Build answer string
        meta = best_match.get("metadata", {})
        session_id = meta.get("source_session", "unknown")
        
        # Grounded description
        answer = f"The {label} is located at coordinate (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}). "
        answer += f"It was promoted during session '{session_id}' with {best_match.get('confidence')*100:.1f}% confidence."

        return {
            "answer": answer,
            "chain_of_thought": steps,
            "provenance": provenance_nodes,
            "confidence": float(confidence)
        }
