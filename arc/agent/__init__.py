"""The agent loop.

perceive → retrieve → plan → act → observe → store, with two dispatch paths behind one
control flow: native tool calling where the model supports it, and a prompted ReAct
protocol with tolerant parsing where it does not (§4.1). The second path is what will
carry a model trained from scratch.
"""

from arc.agent.executor import Executor, Observation
from arc.agent.loop import Agent, AgentResult, Step
from arc.agent.parser import ParsedCall, parse_tool_call

__all__ = [
    "Agent",
    "AgentResult",
    "Executor",
    "Observation",
    "ParsedCall",
    "Step",
    "parse_tool_call",
]
