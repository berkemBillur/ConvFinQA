from __future__ import annotations

"""Lightweight prompt/response tracer for CrewAI executions."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import logging

# Use runtime import guard – typing only
try:
    from crewai import Agent  # type: ignore
except ImportError:  # pragma: no cover
    from typing import Any as Agent  # type: ignore

_logger = logging.getLogger(__name__)


class JsonlTracer:
    """Persist each agent interaction to a JSONL file for later inspection."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")
        
        # Track execution context for parent-child relationships
        self._execution_id = None
        self._current_stage = None
        self._parent_task = None

    # ---------------------------------------------------------------------
    # CrewAI hooks
    # ---------------------------------------------------------------------
    def before_agent(self, agent: Agent, task: str, **kwargs):  # noqa: D401
        self._start_time = time.time()
        self._prompt = task

    def after_agent(self, agent: Agent, response: str, usage: Dict[str, int] | None = None, **kwargs):  # noqa: D401
        record = {
            "ts": time.time(),
            "agent_role": agent.role if hasattr(agent, "role") else str(agent),
            "prompt": self._prompt,
            "response": response,
            "latency": time.time() - self._start_time,
            "usage": usage or {},
            # Enhanced tracking fields
            "execution_id": self._execution_id,
            "stage": self._current_stage,
            "parent_task": self._parent_task,
            "agent_tier": self._get_agent_tier(agent.role if hasattr(agent, "role") else str(agent)),
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
    
    def _get_agent_tier(self, agent_role: str) -> str:
        """Determine agent tier based on role."""
        role_lower = agent_role.lower()
        
        if "conversation manager" in role_lower or "manager" in role_lower:
            return "tier_0"
        elif "extraction specialist" in role_lower or "calculation reasoner" in role_lower:
            return "tier_1"
        elif "critic" in role_lower:
            return "tier_2"
        elif "synthesiser" in role_lower or "synthesis" in role_lower:
            return "tier_3"
        else:
            return "unknown"
    
    # ---------------------------------------------------------------------
    # Enhanced tracking methods
    # ---------------------------------------------------------------------
    def set_execution_context(self, execution_id: str, stage: str, parent_task: str | None = None):
        """Set the current execution context for parent-child tracking."""
        self._execution_id = execution_id
        self._current_stage = stage
        self._parent_task = parent_task
    
    def log_stage_transition(self, from_stage: str, to_stage: str, metadata: Dict[str, Any] | None = None):
        """Log stage transitions in the workflow."""
        record = {
            "ts": time.time(),
            "event_type": "stage_transition",
            "from_stage": from_stage,
            "to_stage": to_stage,
            "execution_id": self._execution_id,
            "metadata": metadata or {},
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
    
    def log_workflow_start(self, workflow_type: str, question: str):
        """Log the start of a workflow execution."""
        self._execution_id = f"{workflow_type}_{int(time.time() * 1000)}"
        record = {
            "ts": time.time(),
            "event_type": "workflow_start",
            "workflow_type": workflow_type,
            "execution_id": self._execution_id,
            "question": question[:200],  # Truncate long questions
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
    
    def log_workflow_end(self, final_answer: str, success: bool = True):
        """Log the end of a workflow execution."""
        record = {
            "ts": time.time(),
            "event_type": "workflow_end",
            "execution_id": self._execution_id,
            "final_answer": final_answer[:200],  # Truncate long answers
            "success": success,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()

    # CrewAI may call __call__ on callback manager; provide noop for safety
    def __call__(self, *args, **kwargs):  # pragma: no cover
        pass


# Convenience helper

def build_tracer(results_dir: Path | None = None) -> JsonlTracer | None:
    if results_dir is None:
        return None
    tracer_path = Path(results_dir) / f"trace_{int(time.time())}.jsonl"
    _logger.info(f"Prompts will be traced to {tracer_path}")
    return JsonlTracer(tracer_path) 