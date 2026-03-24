"""
Agent graph entry point — backward-compatible wrapper.
"""

from agents.supervisor import SupervisorAgent
from utils.llm import get_llm


def create_agent_graph():
    llm = get_llm()
    supervisor = SupervisorAgent(llm)
    return supervisor
