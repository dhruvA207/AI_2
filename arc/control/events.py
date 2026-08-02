"""A marker ARC stamps on the input events it generates.

The kill box in ``overlay.py`` watches the keyboard globally so the user can type the
abort phrase without focusing anything. That watcher sees *every* key, including the
ones ARC itself synthesises — so an agent typing the phrase into a document would
trigger its own abort. Rare, but the failure is silent and confusing, and the fix is
cheap: tag outgoing events here, ignore tagged events there.

``kCGEventSourceUserData`` is a 64-bit field macOS carries on an event and hands back
untouched. Nothing else on the system writes it, so a distinctive value is a reliable
"this one is ours".
"""

from __future__ import annotations

#: Arbitrary but distinctive. Stamped on every synthetic event ARC posts.
ARC_EVENT_TAG = 0x41524301  # "ARC\x01"
