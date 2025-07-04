from __future__ import annotations

"""Factory functions for creating CrewAI agents for the paper replication.

This module implements the 3-agent architecture from "Enhancing Financial Question 
Answering with a Multi-Agent Reflection Framework" (arXiv:2410.21741):

1. Financial Expert Agent: Unified extraction + calculation reasoning
2. Extraction Critic Agent: Reviews data extraction quality  
3. Calculation Critic Agent: Reviews mathematical reasoning

All agent configurations are loaded from config/base.json under the `three_agent_config`
section (configurable). This maintains backward compatibility with the existing predictor
interface.
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

from ...utils.config import Config, APIKeyManager

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory (maintains backward compatibility)
# ---------------------------------------------------------------------------

def build_agents(config: Config) -> Dict[str, Agent]:
    """Build and return the paper's 3-agent architecture for financial QA.

    Implements the multi-agent reflection framework from the paper:
    - Expert: Unified extraction + calculation with chain-of-thought reasoning
    - Extraction Critic: Reviews data extraction accuracy and relevance
    - Calculation Critic: Reviews mathematical reasoning and calculation logic

    All configuration values are loaded from config/base.json under 'multi_agent_paper_v1'.
    Falls back to 'crewai' config for backward compatibility.

    Returns
    -------
    Dict[str, Agent]
        Keys: "expert", "extraction_critic", "calculation_critic"
        
    Raises
    ------
    ValueError
        If required configuration values are missing
    """
    # Try paper config first, fallback to existing config
    agent_cfg = config.get("three_agent_config", {})
    if not agent_cfg:
        _logger.warning("three_agent_config config not found, falling back to crewai config")
        agent_cfg = config.get("crewai", {})
    
    if not agent_cfg:
        raise ValueError(
            "Missing 'three_agent_config' or 'crewai' configuration section in config/base.json. "
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
        value = agent_cfg.get(key)
        if value is None:
            raise ValueError(
                f"Missing '{key}' in config multi_agent_paper_v1 section. "
                f"Required for {agent_name} agent configuration."
            )
        return str(value)
    
    def _get_float_config(key: str, agent_name: str) -> float:
        value = agent_cfg.get(key)
        if value is None:
            raise ValueError(
                f"Missing '{key}' in config multi_agent_paper_v1 section. "
                f"Required for {agent_name} agent configuration."
            )
        return float(value)

    # Paper's Financial Expert Agent (replaces extractor + calculator)
    expert = Agent(
        role='Financial Analysis Expert',
        goal='Extract relevant financial data and perform precise calculations in one unified reasoning process',
        backstory="""You are a financial analysis expert specializing in interpreting earnings reports and financial statements. 
        Your task is to answer specific financial questions based on the given context from financial reports.

        CRITICAL DATA EXTRACTION RULES:
        1. CONVERSATIONAL CONTEXT: Track references like "it", "that", "this value" - they refer to the MOST RECENT calculation result
        2. EXACT METRIC MATCHING: Look for EXACT wording matches first (e.g., "repairs and maintenance" ≠ "maintenance")
        3. TABLE SCANNING: Scan ALL rows and columns systematically - don't stop at first match
        4. YEAR DISAMBIGUATION: When years appear (2012, 2014), distinguish between:
           - Year labels (column headers, time periods)
           - Actual numeric values (prices, amounts)
        5. SIGN PRESERVATION: Maintain correct signs for negative values, changes, and differences

        CALCULATION METHODOLOGY:
        1. PERCENTAGE CHANGE: (new_value - old_value) / old_value
        2. DIFFERENCES: Be careful with order - "change from A to B" = B - A
        3. RATIOS: numerator / denominator (check question for correct order)
        4. PERCENTAGE CONVERSION: Only convert to decimal (÷100) if result should be decimal format
        5. SIGN HANDLING: Preserve negative signs for declines, losses, reductions

        CONVERSATIONAL REFERENCE RESOLUTION:
        - "that" / "it" / "this value" → Use the most recent calculated result
        - "the previous year" → Identify the specific year from context
        - "what is that less 1" → subtract(previous_result, 1)
        - "what is that divided by 100" → divide(previous_result, 100)

        OUTPUT FORMAT:
        Return **only** valid JSON with this exact format:
        {
            "steps": [
                "Step 1: Extract [metric name] for [year]: [value] from [location]",
                "Step 2: Extract [metric name] for [year]: [value] from [location]", 
                "Step 3: Calculate [operation]: [formula] = [result]"
            ],
            "answer": "[numeric_value_only]"
        }

        ANSWER FIELD RULES:
        - Return ONLY the numeric value (e.g., "0.7893" not "78.93")
        - No units, symbols, or text (e.g., "124" not "124 million")
        - Preserve all decimal places from calculations
        - Maintain negative signs where appropriate
        """,
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("expert_model", "expert"),
                 _get_float_config("expert_temperature", "expert")),
        tools=[],  # Pure reasoning approach as per paper
    )

    # Paper's Extraction Critic Agent
    extraction_critic = Agent(
        role='Data Extraction Quality Critic',
        goal='Review and critique the accuracy and relevance of extracted financial data',
        backstory="""You are a data extraction quality specialist who reviews financial 
        analysis work with a focus on data accuracy and relevance.
        
        When reviewing an expert's response, you specifically evaluate:
        - Are the extracted numbers actually relevant to the question being asked?
        - Were any important data points missed or overlooked?
        - Are the numbers correctly interpreted from the source tables and text?
        - Is the data extraction reasoning sound and well-justified?
        - Are there any ambiguities in data interpretation that need clarification?
        
        Respond with JSON:

        {{"is_correct": <true|false>,
        "issues": ["<issue 1>", …],
        "suggested_fix": "<concise suggestion if false>"}}""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("extraction_critic_model", "extraction_critic"),
                 _get_float_config("extraction_critic_temperature", "extraction_critic")),
        tools=[],  # Pure reasoning approach
    )

    # Paper's Calculation Critic Agent  
    calculation_critic = Agent(
        role='Financial Calculation Logic Critic',
        goal='Review and critique the mathematical reasoning and calculation methods used in financial analysis',
        backstory="""You are a financial calculation logic specialist who reviews mathematical 
        reasoning in financial analysis with expertise in financial calculations and business logic.
        
        When reviewing an expert's response, you specifically evaluate:
        - Is the calculation method appropriate for the type of question being asked?
        - Are the mathematical operations performed correctly and in the right order?
        - Does the final result magnitude make sense in the business/financial context?
        - Is the step-by-step reasoning logically sound and easy to follow?
        - Are there any errors in formula application or computational logic?
        - Are units, scales, and percentages handled correctly?
        
        You provide specific, actionable feedback that helps improve calculation accuracy 
        and logical reasoning. Output JSON:

        {{"is_correct": <true|false>,
        "issues": ["<issue 1>", …],
        "suggested_fix": "<concise suggestion if false>"}}""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("calculation_critic_model", "calculation_critic"),
                 _get_float_config("calculation_critic_temperature", "calculation_critic")),
        tools=[],  # Pure reasoning approach
    )

    return {
        "expert": expert,
        "extraction_critic": extraction_critic,
        "calculation_critic": calculation_critic,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_api_key(config: Config):
    """Try multiple places to find an OpenAI API key."""
    key = APIKeyManager.load_openai_key()
    if key:
        return key
    
    # Fallback to config object if provided
    key = config.get("openai_api_key")
    if key:
        _logger.debug("✅ API key loaded from config object")
        return key
    
    return None 