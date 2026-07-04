from typing import Dict, Any, List

class AccessController:
    def __init__(self):
        # Role authorizations
        self.role_privileges = {
            "admin": ["read_raw", "write_raw", "clear_all", "export"],
            "operator": ["read_raw", "write_raw", "export"],
            "robot-runtime": ["read_raw", "write_raw"],
            "guest-query": ["read_raw"]
        }

    def verify_privilege(self, role: str, privilege: str) -> bool:
        """
        Verify that a role has the requested permissions before routing operations.
        """
        privs = self.role_privileges.get(role, [])
        return privilege in privs
