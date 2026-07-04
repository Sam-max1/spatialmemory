import time
import json
from typing import Dict, List, Any
from pathlib import Path

class ImprovementLog:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._init_log()

    def _init_log(self) -> None:
        if not self.log_path.exists():
            with open(self.log_path, "w") as f:
                f.write("")

    def append_entry(self, learning_type: str, evidence: str, effect: str) -> None:
        """
        Record a learning event to the append-only ledger.
        """
        entry = {
            "timestamp": time.time(),
            "learning_type": learning_type,
            "evidence": evidence,
            "effect": effect
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_entries(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        
        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        return entries
