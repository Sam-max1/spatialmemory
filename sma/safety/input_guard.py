from typing import Dict, Any

class InputGuard:
    def __init__(self):
        # A simple blacklist of prompt injection substrings
        self.prompt_injection_blacklist = [
            "ignore previous instructions",
            "system override",
            "sudo",
            "format c:",
            "rm -rf"
        ]

    def screen_query(self, query_text: str) -> bool:
        """
        Check query for adversarial prompt-injection or control hijacking attempts.
        Returns True if safe, False if unsafe/injection suspected.
        """
        txt = query_text.lower()
        for pattern in self.prompt_injection_blacklist:
            if pattern in txt:
                return False
        return True

    def screen_image_artifacts(self, image_data: bytes) -> bool:
        """
        Detect visual anomalies (e.g., printed adversarial textures or QR-code instructions)
        that could spoof Place Recognition descriptors.
        """
        # In a real environment, checks are run using anomalous texture classifiers.
        # We simulate visual screen safety.
        if b"ADVERSARIAL_PATCH" in image_data:
            return False
        return True
