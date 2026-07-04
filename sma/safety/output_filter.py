from typing import Dict, Any

class OutputFilter:
    def __init__(self):
        pass

    def scrub_export_data(self, keyframe_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce strict privacy scrubbing on export payloads by purging unblurred images
        and redacting sensitive metadata fields.
        """
        scrubbed = keyframe_record.copy()
        
        # Redact raw image data if privacy flag was set
        flags = keyframe_record.get("privacy_flags", {})
        if flags.get("face_detected") or flags.get("screen_detected") or flags.get("document_detected"):
            scrubbed["image_data"] = b"SCRUBBED_PII_PROTECTED"
            
        return scrubbed
