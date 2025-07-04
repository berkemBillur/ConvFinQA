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
from .tools.calculation_tools import CalculationTool

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
        backstory="""You are a data extraction quality specialist.

EVALUATION CRITERIA:
1. RELEVANCE – Are extracted numbers relevant to the question?
2. COMPLETENESS – Were any required data points missed?
3. ACCURACY – Are row/column labels and raw values correct?
4. SCALING – Are units and scale multipliers handled properly?
5. TEMPORAL – Are correct years / time periods selected?
6. REFERENCES – Are conversational references resolved?

OUTPUT FORMAT:
{
  "is_correct": <true|false>,
  "issues": ["issue 1", …],
  "suggested_fix": "concise suggestion if false, empty string if true"
}

Guidelines:
• Only `is_correct: true` if extraction is fully accurate and complete
• Provide specific actionable issues and fixes when false
""",
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
        backstory="""You are a financial calculation logic specialist.

EVALUATION CRITERIA:
1. METHOD – Is the calculation method appropriate for the question type?
2. OPERATIONS – Are mathematical operations performed correctly & in correct order?
3. LOGIC – Is step-by-step reasoning sound?
4. SCALE – Are unit conversions / multipliers applied correctly?
5. BUSINESS SENSE – Does result magnitude make sense in context?
6. DSL – If DSL provided, is syntax & logic correct?

OUTPUT FORMAT:
{
  "is_correct": <true|false>,
  "issues": ["issue 1", …],
  "suggested_fix": "concise suggestion if false, empty string if true"
}

Guidelines:
• Mark `is_correct: true` **only** when calculation logic is entirely correct
• Provide specific actionable fixes otherwise
""",
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

def build_agents_six(config: Config) -> Dict[str, Agent]:
    """Build and return the six-agent architecture for financial QA.

    Implements the tiered six-agent framework:
    - Tier 0: Conversation Manager (routing + cache)
    - Tier 1: Data Extraction Specialist + Fin-Calc Reasoner
    - Tier 2: Extraction Critic + Calculation Critic  
    - Tier 3: Answer Synthesiser

    All configuration values are loaded from config/base.json under 'six_agent_config'.

    Returns
    -------
    Dict[str, Agent]
        Keys: "manager", "extractor", "reasoner", "extraction_critic", 
              "calculation_critic", "synthesiser"
        
    Raises
    ------
    ValueError
        If required configuration values are missing
    """
    agent_cfg = config.get("six_agent_config", {})
    if not agent_cfg:
        raise ValueError(
            "Missing 'six_agent_config' configuration section in config/base.json. "
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
                f"Missing '{key}' in config six_agent_config section. "
                f"Required for {agent_name} agent configuration."
            )
        return str(value)
    
    def _get_float_config(key: str, agent_name: str) -> float:
        value = agent_cfg.get(key)
        if value is None:
            raise ValueError(
                f"Missing '{key}' in config six_agent_config section. "
                f"Required for {agent_name} agent configuration."
            )
        return float(value)

    # Tier 0: Conversation Manager
    manager = Agent(
        role='Conversation Manager',
        goal='Route questions efficiently and cache answers to reduce latency and cost',
        backstory="""You manage the conversation and decide whether to use a cached answer or run the full pipeline.

Steps you follow:
1. Check if the same question is already in the cache.
2. If the question is a simple follow-up (for example "same as above?") see if the cache has the needed value.
3. If a clear answer is found in cache return it; otherwise start the six-agent pipeline.
4. Add a short note explaining your decision so other agents can audit it later.

Output JSON:
{
  "action": "cache_hit" | "run_pipeline",
  "cached_answer": "<answer_if_any>",
  "reasoning": "<why_you_chose_this_action>"
}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("manager_model", "manager"),
                 _get_float_config("manager_temperature", "manager")),
        tools=[],
    )

    # Tier 1: Data Extraction Specialist
    extractor = Agent(
        role='Data Extraction Specialist',
        goal='Isolate exact table cells and normalize units, scale, and polarity',
        backstory="""You are a financial data expert that extracts the exact numbers needed from the financial document.

Steps you follow:
1. Read the question and understand what metric or period or relevant information it asks for.
2. If a table exists, scan all table rows and columns for exact wording first, then close matches.
3. Note unit hints like "in millions" or "in billions" and convert parentheses to minus signs.
4. Resolve conversational references such as "that value" or "previous year".
5. List each extraction clearly.

Output JSON:
{
  "extractions": [
    {"row": "<row_label>", "col": "<column_label>", "raw": "<text_value>", "unit": "<unit>", "scale": <multiplier>}
  ],
  "references_resolved": ["<explanation>"] ,
  "extraction_notes": "<optional_note>"
}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("extractor_model", "extractor"),
                 _get_float_config("extractor_temperature", "extractor")),
        tools=[],
    )

    # Tier 1: Fin-Calc Reasoner
    reasoner = Agent(
        role='Financial Calculation Reasoner',
        goal='Perform chain-of-thought arithmetic and generate executable DSL programs',
        backstory="""You are a fincncial calculation specialist that performs the calculations that answer the question.

Steps you follow:
1. Review the extractor JSON and pick the needed numbers.
2. Decide which formula applies: percentage change, difference, ratio, or simple lookup.
3. Show each step of the math using plain language.
4. When the math is complex, produce a short DSL string so it can be checked later.
5. Give the final numeric answer with no extra text.

Output JSON:
{
  "steps": ["Step 1: …"],
  "dsl": "<optional_dsl_program>",
  "answer": "<numeric_value_only>"
}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("reasoner_model", "reasoner"),
                 _get_float_config("reasoner_temperature", "reasoner")),
        tools=[CalculationTool()],
    )

    # Tier 2: Extraction Critic
    extraction_critic = Agent(
        role='Data Extraction Quality Critic',
        goal='Verify extraction accuracy: correct rows, right fiscal year, proper scaling',
        backstory="""You check the extractor's work.

Steps you follow:
1. Compare each extracted number with the document.
2. Check that rows, columns, units, and years are correct.
3. List any missing or wrong values.
4. Decide if the extraction is fully correct.

Output JSON:
{
  "is_correct": true | false,
  "issues": ["<issue1>", "<issue2>"] ,
  "suggested_fix": "<short_suggestion_if_needed>"
}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("extraction_critic_model", "extraction_critic"),
                 _get_float_config("extraction_critic_temperature", "extraction_critic")),
        tools=[],
    )

    # Tier 2: Calculation Critic
    calculation_critic = Agent(
        role='Financial Calculation Logic Critic',
        goal='Verify order of operations, unit consistency, and magnitude sanity',
        backstory="""You check the reasoner's calculations.

Steps you follow:
1. Read each calculation step and formula.
2. Verify numbers, order of operations, and unit conversions.
3. Check whether the result size makes sense in business context.
4. If a DSL program is given, check its syntax and logic.
5. Decide if the calculation is correct.

Output JSON:
{
  "is_correct": true | false,
  "issues": ["<issue1>", "<issue2>"] ,
  "suggested_fix": "<short_suggestion_if_needed>"
}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("calculation_critic_model", "calculation_critic"),
                 _get_float_config("calculation_critic_temperature", "calculation_critic")),
        tools=[],
    )

    # Tier 3: Answer Synthesiser
    synthesiser = Agent(
        role='Answer Synthesiser',
        goal='Merge reasoner output with critic feedback and format final conversational reply',
        backstory="""You write the final answer or ask for a revision.

Steps you follow:
1. Read the reasoner output and both critic JSON blocks.
2. If both critics say is_correct is true, take the numeric answer.
3. Clean the number: remove commas, keep sign and decimals.
4. Write a short friendly sentence with the number.
5. If any critic says false, list the main problems in 2–3 short bullet points and ask for a revision.

Output JSON (choose one):
1. Final answer:
   {"status": "final", "answer": "<number>"}
2. Revision needed:
   {"status": "revise", "critique_summary": "<bullet_point_list>"}
""",
        allow_delegation=False,
        verbose=agent_cfg.get("verbose", True),
        llm=_llm(_get_string_config("synthesiser_model", "synthesiser"),
                 _get_float_config("synthesiser_temperature", "synthesiser")),
        tools=[],
    )

    return {
        "manager": manager,
        "extractor": extractor,
        "reasoner": reasoner,
        "extraction_critic": extraction_critic,
        "calculation_critic": calculation_critic,
        "synthesiser": synthesiser,
    }


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