from typing import Dict, Any, Tuple

class IntentClassifier:
    def __init__(self):
        pass

    def classify_intent(self, query_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify user query into system intents:
        - locate-object (e.g. "where is the fire extinguisher?")
        - navigate-to (e.g. "go to the kitchen")
        - describe-region (e.g. "what is in this office?")
        - session-control (e.g. "start mapping")
        """
        txt = query_text.lower()
        params = {}
        
        if "where" in txt or "locate" in txt or "find" in txt:
            # Locate object intent
            # Extract target object label
            label = "unknown"
            words = txt.split()
            for i, w in enumerate(words):
                if w in ["where", "locate", "find"] and i + 2 < len(words) and words[i+1] == "is":
                    label = " ".join(words[i+2:])
                    break
                elif w in ["find", "locate"] and i + 1 < len(words):
                    label = " ".join(words[i+1:])
                    break
            
            # Clean up label (strip punctuation and leading articles)
            label = label.replace("?", "").replace(".", "").strip()
            for article in ["the ", "a ", "an "]:
                if label.startswith(article):
                    label = label[len(article):].strip()
            params["label"] = label
            return "locate-object", params
            
        elif "go to" in txt or "navigate" in txt or "drive" in txt or "walk" in txt:
            destination = "hallway"
            if "go to" in txt:
                destination = txt.split("go to")[-1].strip()
            elif "navigate to" in txt:
                destination = txt.split("navigate to")[-1].strip()
            # Clean up destination (strip punctuation and leading articles)
            destination = destination.replace(".", "").strip()
            for article in ["the ", "a ", "an "]:
                if destination.startswith(article):
                    destination = destination[len(article):].strip()
            params["destination"] = destination
            return "navigate-to", params
            
        elif "describe" in txt or "what is in" in txt or "summarize" in txt:
            # Describe region intent
            params["radius"] = 3.0
            return "describe-region", params
            
        elif "start" in txt or "stop" in txt or "resume" in txt or "close" in txt:
            # Session control intent
            if "start" in txt:
                params["action"] = "start"
            elif "stop" in txt or "close" in txt:
                params["action"] = "stop"
            return "session-control", params
            
        return "describe-region", params
