"""Append-only action logging and the kill switch.

Both exist for the reason the brief gives in §0.3: they are not guardrails, they are
the instruments you need to debug an agent that has unrestricted control of the
machine. The audit log answers "what did it do at 2am"; the kill switch answers "how
do I stop it right now".
"""

from arc.audit.killswitch import KillSwitch
from arc.audit.logger import AuditLogger, AuditRecord

__all__ = ["AuditLogger", "AuditRecord", "KillSwitch"]
