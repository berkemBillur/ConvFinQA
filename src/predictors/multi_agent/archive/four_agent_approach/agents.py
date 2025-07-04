from __future__ import annotations

"""Factory functions for creating CrewAI agents used in ConvFinQA.

This module purposefully contains *no* CrewAI-specific workflow code – it only
instantiates `crewai.Agent` objects wired with the correct LLM models and
specialised tools.  Keeping this logic separate simplifies testing and future
changes to agent definitions.

All agent configurations (models, temperatures, etc.) are loaded from config/base.json.
No hardcoded values are used - everything is configurable via the config system.
"""

from typing import Dict

import logging

try:
    from crewai import Agent
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr
except ImportError as exc:  # pragma: no cover – handled by caller
    raise ImportError(
        "CrewAI and langchain_openai must be installed to build agents."
    ) from exc

from ..tools.extraction_tools import (
    TableExtractionTool,
    ReferenceResolverTool,
    TemporalParserTool,
)
from ..tools.supervisor_tools import TaskDecomposerTool, ConversationTrackerTool
from ...utils.config import Config

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_agents(config: Config) -> Dict[str, Agent]:
    """Build and return the four ConvFinQA CrewAI agents.

    All configuration values are loaded from config/base.json under the 'crewai' section.
    No hardcoded fallbacks are used to ensure configuration transparency.

    Returns
    -------
    Dict[str, Agent]
        Keys: "supervisor", "extractor", "calculator", "validator".
        
    Raises
    ------
    ValueError
        If required configuration values are missing from config/base.json
    """
    crew_cfg = config.get("crewai", {})
    
    if not crew_cfg:
        raise ValueError(
            "Missing 'crewai' configuration section in config/base.json. "
            "Please ensure all agent model and temperature settings are defined."
        )

    # Helper to create ChatOpenAI with graceful fallback for missing API key
    api_key = _load_api_key(config)

    def _llm(model: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=SecretStr(api_key) if api_key else None,
        )

    # Get configuration values with clear error messages if missing
    def _get_string_config(key: str, agent_name: str) -> str:
        value = crew_cfg.get(key)
        if value is None:
            raise ValueError(
                f"Missing '{key}' in config/base.json crewai section. "
                f"Required for {agent_name} agent configuration."
            )
        return str(value)
    
    def _get_float_config(key: str, agent_name: str) -> float:
        value = crew_cfg.get(key)
        if value is None:
            raise ValueError(
                f"Missing '{key}' in config/base.json crewai section. "
                f"Required for {agent_name} agent configuration."
            )
        return float(value)

    supervisor = Agent(
        role='Financial QA Orchestrator',
        goal='Decompose conversational financial queries and coordinate specialist agents for accurate answers',
        backstory="""You are a senior financial analyst supervisor who excels at breaking down 
        complex conversational finance questions into structured subtasks. You maintain conversation 
        context across multiple turns and coordinate specialist agents to produce accurate, 
        well-reasoned answers.""",
        allow_delegation=False,  # Disable delegation to avoid tool format issues
        verbose=crew_cfg.get("verbose", True),
        llm=_llm(_get_string_config("supervisor_model", "supervisor"),
                 _get_float_config("supervisor_temperature", "supervisor")),
        tools=[],  # Manager agent must not carry tools
    )

    extractor = Agent(
        role='Financial Data Extraction Specialist',
        goal='Extract precise numerical data from financial documents and resolve conversational references',
        backstory="""You are a data extraction expert who specialises in financial documents. 
        You excel at finding specific numerical values in complex tables and resolving 
        conversational references like 'it', 'that year', and 'the previous quarter' across 
        multi-turn conversations.""",
        allow_delegation=False,
        verbose=crew_cfg.get("verbose", True),
        llm=_llm(_get_string_config("extractor_model", "extractor"),
                 _get_float_config("extractor_temperature", "extractor")),
        tools=[],  # Simplify: no explicit tool calls, agent extracts directly from prompt
    )

    calculator = Agent(
        role='Financial Calculations Specialist',
        goal='Perform accurate financial calculations and generate executable DSL programs',
        backstory="""You are a quantitative financial analyst who performs complex calculations 
        and generates precise DSL programs. You understand financial business logic, apply 
        appropriate calculation methods, and ensure mathematical accuracy in financial contexts.""",
        allow_delegation=False,
        verbose=crew_cfg.get("verbose", True),
        llm=_llm(_get_string_config("calculator_model", "calculator"),
                 _get_float_config("calculator_temperature", "calculator")),
        tools=[],  # No tool usage; agent must respond directly without calls
    )

    validator = Agent(
        role='Financial QA Validator',
        goal='Validate answers through cross-agent verification and confidence scoring',
        backstory="""You are a financial QA validator who performs final verification of answers. 
        You check for logical consistency, numerical accuracy, and conversational context correctness. 
        You provide confidence scores and identify potential errors before final answer delivery.""",
        allow_delegation=False,
        verbose=crew_cfg.get("verbose", True),
        llm=_llm(_get_string_config("validator_model", "validator"),
                 _get_float_config("validator_temperature", "validator")),
        tools=[],  # Validation handled via reasoning only; no external tools
    )

    return {
        "supervisor": supervisor,
        "extractor": extractor,
        "calculator": calculator,
        "validator": validator,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_api_key(config: Config):
    """Try multiple places to find an OpenAI API key."""
    import os
    key = os.getenv("OPENAI_API_KEY")
    if key and key != "sk-your-openai-api-key-here":
        _logger.debug("✅ API key loaded from environment")
        return key
    return config.get("openai_api_key") 