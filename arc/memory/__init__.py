"""Persistent memory.

Three cooperating layers over one SQLite file (§4.2):

- **episodic** — timestamped events and conversation turns. "What did we do last
  Tuesday?"
- **semantic** — facts and entities with typed relationships, stored as a graph.
  "Who is X and how do they relate to Y?", including multi-hop.
- **procedural** — learned workflows and preferences, written by consolidation out of
  repeated episodic patterns.

Retrieval is hybrid across all three, and consolidation runs in the background to
dedupe, summarise, decay, and promote. Nothing here talks to a network.
"""

from arc.memory.embedder import Embedder, HashEmbedder
from arc.memory.store import MemoryRecord, MemoryStore

__all__ = ["Embedder", "HashEmbedder", "MemoryRecord", "MemoryStore"]
