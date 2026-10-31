"""SentinelOps LangGraph state machine.

    triage ─► db_inspector ◄─┐ (self-correction: invalid args / tool failure, ≤ MAX_TOOL_RETRIES)
                  │ ▲        │
                  ▼ └────────┘
          policy_retriever
                  │
                  ▼
        resolution_planner ──(needs older ledger history)──► db_inspector
                  │
     ┌────────────┴──────────────┐
     ▼ (> $200 / high risk /      ▼ (≤ $200 & low risk)
   human_gate  injection)         │
     │ approved   │ rejected      │
     ▼            └───────┐       │
  action_executor ◄───────┼───────┘
     │                    │
     ▼                    ▼
  response_drafter ◄── llm_judge (score < 0.85 → re-draft)
                           │ (score ≥ 0.85)
                           ▼
                       dispatch ─► END
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import AgentDeps, SentinelNodes
from app.models.state import AgentState

NODES: tuple[str, ...] = (
    "triage",
    "db_inspector",
    "policy_retriever",
    "resolution_planner",
    "human_gate",
    "action_executor",
    "response_drafter",
    "llm_judge",
    "dispatch",
)

EDGES: tuple[tuple[str, str, str], ...] = (
    ("__start__", "triage", "always"),
    ("triage", "db_inspector", "always"),
    ("db_inspector", "db_inspector", "self-correction retry"),
    ("db_inspector", "policy_retriever", "ok / retries exhausted"),
    ("policy_retriever", "resolution_planner", "always"),
    ("resolution_planner", "db_inspector", "re-query wider window"),
    ("resolution_planner", "human_gate", "> $200 / high risk"),
    ("resolution_planner", "action_executor", "≤ $200 & low risk"),
    ("human_gate", "action_executor", "approved"),
    ("human_gate", "response_drafter", "rejected"),
    ("action_executor", "response_drafter", "always"),
    ("response_drafter", "llm_judge", "always"),
    ("llm_judge", "response_drafter", "score < 0.85"),
    ("llm_judge", "dispatch", "score ≥ 0.85"),
    ("dispatch", "__end__", "always"),
)


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[str] | None = None) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    n = SentinelNodes(deps)
    s = deps.settings

    def route_after_db(state: AgentState) -> Literal["db_inspector", "policy_retriever"]:
        if state.get("last_tool_error") and state["tool_error_count"] <= s.max_tool_retries:
            return "db_inspector"
        return "policy_retriever"

    def route_after_planner(state: AgentState) -> Literal["db_inspector", "human_gate", "action_executor"]:
        if state.get("requery_requested"):
            return "db_inspector"
        return "human_gate" if state.get("requires_human_approval") else "action_executor"

    def route_after_gate(state: AgentState) -> Literal["action_executor", "response_drafter"]:
        return "action_executor" if state.get("human_approved") else "response_drafter"

    def route_after_judge(state: AgentState) -> Literal["response_drafter", "dispatch"]:
        verdict = state.get("judge_verdict") or {}
        if verdict.get("passed"):
            return "dispatch"
        return "response_drafter" if state.get("draft_attempts", 0) <= s.max_redrafts else "dispatch"

    g: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    g.add_node("triage", n.triage)
    g.add_node("db_inspector", n.db_inspector)
    g.add_node("policy_retriever", n.policy_retriever)
    g.add_node("resolution_planner", n.resolution_planner)
    g.add_node("human_gate", n.human_gate)
    g.add_node("action_executor", n.action_executor)
    g.add_node("response_drafter", n.response_drafter)
    g.add_node("llm_judge", n.llm_judge)
    g.add_node("dispatch", n.dispatch)

    g.add_edge(START, "triage")
    g.add_edge("triage", "db_inspector")
    g.add_conditional_edges("db_inspector", route_after_db)
    g.add_edge("policy_retriever", "resolution_planner")
    g.add_conditional_edges("resolution_planner", route_after_planner)
    g.add_conditional_edges("human_gate", route_after_gate)
    g.add_edge("action_executor", "response_drafter")
    g.add_edge("response_drafter", "llm_judge")
    g.add_conditional_edges("llm_judge", route_after_judge)
    g.add_edge("dispatch", END)
    return g.compile(checkpointer=checkpointer)
