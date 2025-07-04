from __future__ import annotations

"""Task templating utilities for paper replication multi-agent system.

This module implements the task structure from "Enhancing Financial Question 
Answering with a Multi-Agent Reflection Framework" (arXiv:2410.21741):

1. Expert Task: Unified extraction + calculation with JSON output
2. Extraction Critic Task: Reviews data extraction quality
3. Calculation Critic Task: Reviews mathematical reasoning

The tasks use pure prompt-based reasoning without tools, matching the paper's approach.
"""

from dataclasses import dataclass
from typing import List, Any
import re

try:
    from crewai import Task
except ImportError as exc:  # pragma: no cover
    raise ImportError("CrewAI must be installed to build tasks.") from exc

from ...data.models import ConvFinQARecord
from ...utils.financial_matcher import financial_matcher
from ...utils.scale_normalizer import scale_normalizer


@dataclass
class Context:
    """Minimal context object shared across task builders."""

    record: ConvFinQARecord
    turn_index: int
    conversation_history: List[dict[str, str]]
    iteration: int = 0  # For tracking refinement iterations
    critic_feedback: str = ""  # For storing critic feedback

    @property
    def question(self) -> str:
        return self.record.dialogue.conv_questions[self.turn_index]

    @property
    def formatted_history(self) -> str:
        if not self.conversation_history:
            return "No previous turns."
        parts: list[str] = []
        for i, t in enumerate(self.conversation_history, 1):
            parts.append(f"Turn {i}: Q: {t['question']} | A: {t['answer']}")
        return "\n".join(parts)

    @property
    def formatted_document(self) -> str:
        # Enhanced formatting with actual table data rows
        doc = self.record.doc
        
        # Create table header
        headers = ["Row Label"] + list(doc.table.keys())
        table_lines = ["\t".join(headers)]
        
        # Add table data rows
        if doc.table:
            # Get all unique row labels from all columns
            all_row_labels = set()
            for column_data in doc.table.values():
                all_row_labels.update(column_data.keys())
            
            # Create a row for each row label
            for row_label in sorted(all_row_labels):
                row_data = [row_label]
                for column_name in doc.table.keys():
                    cell_value = doc.table[column_name].get(row_label, "N/A")
                    row_data.append(str(cell_value))
                table_lines.append("\t".join(row_data))
        
        table_str = "\n".join(table_lines)
        return f"PRE-TEXT:\n{doc.pre_text[:1200]}...\n\nTABLE:\n{table_str}"
    
    def extract_target_kpis(self) -> List[str]:
        """Extract potential KPI terms from the question for smart matching."""
        question = self.question.lower()
        
        # Common KPI extraction patterns
        kpi_patterns = [
            r"(?:what (?:was|is|were) (?:the )?)([\w\s&]+?)(?:\s+in|\s+for|\s+during|\s*\?)",
            r"(?:decline in|increase in|change in)\s+([\w\s&]+?)(?:\s+|,|\?)",
            r"(?:ratio of|percentage of)\s+([\w\s&]+?)(?:\s+to|\s+in|\s*\?)",
            r"([\w\s&]+?)\s+(?:expense|expenses|cost|costs|income|revenue|earnings)",
            r"(?:repair|maintenance|repairs and maintenance|r&m)",
        ]
        
        kpis = []
        for pattern in kpi_patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 2:  # Filter out very short matches
                    kpis.append(cleaned)
        
        # Add some common financial terms if mentioned
        financial_keywords = [
            "net earnings", "operating income", "revenue", "repairs and maintenance",
            "depreciation", "assets", "liabilities", "cash flow", "earnings per share"
        ]
        
        for keyword in financial_keywords:
            if keyword in question:
                kpis.append(keyword)
        
        return list(set(kpis))  # Remove duplicates


# ---------------------------------------------------------------------------
# Paper-Style Task Builders
# ---------------------------------------------------------------------------

def build_expert_task(ctx: Context, agent, prev_expert_response: str | None = None) -> Task:
    """Create the unified expert task for extraction + calculation (paper's approach)."""
    
    # Build iteration context
    iteration_context = ""
    if ctx.iteration > 0:
        prev_resp_block = "" if prev_expert_response is None else f"\nPREVIOUS EXPERT RESPONSE (for your reference):\n{prev_expert_response}\n"
        iteration_context = f"""
REFINEMENT ITERATION {ctx.iteration}:
You are revising your previous response based on critic feedback.
Previous critic feedback to address:
{ctx.critic_feedback}

{prev_resp_block}

Please improve your analysis by addressing the specific issues raised by the critics.
"""
    
    desc = f"""
FINANCIAL ANALYSIS EXPERT TASK - Unified Extraction and Calculation

{iteration_context}

QUESTION: "{ctx.question}"

CONVERSATION HISTORY:
{ctx.formatted_history}

FINANCIAL DOCUMENT:
{ctx.formatted_document}

ANALYSIS STRATEGY:

STEP 1 - UNDERSTAND THE QUESTION:
- What specific metric is being asked for?
- What time periods are involved?
- Are there conversational references to resolve?
- What type of calculation is needed (lookup, change, ratio, percentage)?

STEP 2 - RESOLVE CONVERSATIONAL CONTEXT:
- If question contains "that", "it", "this value" → use the most recent result from conversation history
- Example: Previous answer was 78.93, question asks "what is that divided by 100?" → use 78.93
- "the previous year" → identify specific year from context
- "the difference between the two values" → use the two most recently mentioned values

STEP 3 - DATA EXTRACTION:
- Scan the ENTIRE table systematically (all rows, all columns)
- Look for EXACT metric name matches first, then synonyms
- When you see years (2012, 2014), distinguish:
  * Column headers (time periods) vs actual numeric values
  * Don't confuse year labels with the data values
- For ambiguous cases, choose the value that makes business sense for the question

STEP 4 - CALCULATION:
- Percentage change: (new - old) / old
- Differences: pay attention to order ("change from A to B" = B - A)
- Ratios: check question for correct numerator/denominator
- Preserve negative signs for declines, losses, decreases

STEP 5 - VERIFICATION:
- Does the result magnitude make sense?
- Are signs correct (positive/negative)?
- Is the scale appropriate?

COMMON ERROR PATTERNS TO AVOID:
- Don't return years (2012, 2014) when asked for financial values
- Don't lose negative signs in calculations
- Don't confuse different metrics (net earnings ≠ operating income)
- Don't ignore conversational context ("that" refers to previous result)
"""

    expected_output = """JSON format exactly like this example:
{
    "steps": [
        "Extract revenue for 2020: $60.94M from table row 'Total Revenue'",
        "Extract revenue for 2019: $25.14M from table row 'Total Revenue'", 
        "Calculate percentage change: (60.94 - 25.14) / 25.14 = 1.42",
        "Convert to percentage: 1.42 × 100% = 142%"
    ],
    "answer": "142%"
}

Each step should be clear and specific about what data was extracted and what calculations were performed."""

    return Task(
        description=desc,
        agent=agent,
        expected_output=expected_output
    )


def build_extraction_critic_task(ctx: Context, expert_response: str, agent) -> Task:
    """Create extraction critic task to review data extraction quality."""
    
    desc = f"""
DATA EXTRACTION QUALITY CRITIC TASK

Review the expert's response for data extraction accuracy and relevance.

ORIGINAL QUESTION: "{ctx.question}"
DOCUMENT CONTEXT: {ctx.formatted_document}
EXPERT'S RESPONSE TO REVIEW: {expert_response}

EVALUATION CRITERIA:
1. RELEVANCE: Are the extracted numbers actually relevant to the question?
2. COMPLETENESS: Were any important data points missed?
3. ACCURACY: Are the numbers correctly interpreted from the source?
4. CONTEXT: Is the data extraction reasoning sound and well-justified?
5. REFERENCES: Are conversational references properly resolved?

REVIEW INSTRUCTIONS:
- Carefully examine each extracted number in the expert's response
- Check if the numbers directly answer what the question is asking
- Verify the numbers exist in the provided document
- Assess if the extraction reasoning is clear and logical
- Identify any missing data that should have been extracted

Provide your assessment **in EXACTLY the JSON format below and nothing else (no narrative)**:
{{
  "is_correct": <true | false>,
  "issues": ["<short issue 1>", "..."],
  "suggested_fix": "<concise suggestion or empty string>"
}}

Guidelines:
- "is_correct": true only when the extraction is fully accurate and complete; otherwise false.
- When false, list concrete issues in "issues" and propose a concise remedy in "suggested_fix".
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"is_correct": true, "issues": [], "suggested_fix": ""}'
    )


def build_calculation_critic_task(ctx: Context, expert_response: str, agent) -> Task:
    """Create calculation critic task to review mathematical reasoning."""
    
    desc = f"""
CALCULATION LOGIC CRITIC TASK

Review the expert's response for mathematical reasoning and calculation accuracy.

ORIGINAL QUESTION: "{ctx.question}"
EXPERT'S RESPONSE TO REVIEW: {expert_response}

EVALUATION CRITERIA:
1. METHOD: Is the calculation method appropriate for the question type?
2. OPERATIONS: Are mathematical operations performed correctly?
3. LOGIC: Is the step-by-step reasoning logically sound?
4. BUSINESS_SENSE: Does the result magnitude make sense in context?
5. UNITS: Are scales, percentages, and units handled correctly?

REVIEW INSTRUCTIONS:
- Examine the calculation method chosen by the expert
- Verify each mathematical operation is performed correctly
- Check if the calculation sequence is logical and appropriate
- Assess if the final result makes business/financial sense
- Ensure proper handling of percentages, ratios, and financial scales

Common calculation patterns to verify:
- Percentage change: (new - old) / old × 100%
- Growth rates: (current - previous) / previous
- Ratios: numerator / denominator
- Differences: value1 - value2

Provide your assessment **in EXACTLY the JSON format below and nothing else (no narrative)**:
{{
  "is_correct": <true | false>,
  "issues": ["<short issue 1>", "..."],
  "suggested_fix": "<concise suggestion or empty string>"
}}

Guidelines:
- "is_correct": true only when the calculation logic is entirely correct and sound.
- If false, enumerate concrete problems in "issues" and give a brief remedy in "suggested_fix".
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"is_correct": true, "issues": [], "suggested_fix": ""}'
    )


def build_paper_tasks(ctx: Context, agents: dict[str, Any], expert_response: str | None = None, prev_expert_response: str | None = None) -> List[Task]:
    """Create tasks based on the paper's approach.
    
    If expert_response is None, creates initial expert task.
    If expert_response is provided, creates critic tasks to review it.
    """
    if expert_response is None:
        # Expert task (initial or revised)
        expert_task = build_expert_task(ctx, agents["expert"], prev_expert_response=prev_expert_response)
        return [expert_task]
    else:
        # Critic tasks to review expert response
        extraction_critic_task = build_extraction_critic_task(ctx, expert_response, agents["extraction_critic"])
        calculation_critic_task = build_calculation_critic_task(ctx, expert_response, agents["calculation_critic"])
        return [extraction_critic_task, calculation_critic_task]


# ---------------------------------------------------------------------------
# Legacy compatibility functions (for backward compatibility)
# ---------------------------------------------------------------------------

def build_tasks(ctx: Context, agents: dict[str, Any]) -> List[Task]:
    """Legacy function for backward compatibility. 
    
    This now creates the initial expert task using the paper's approach.
    """
    return build_paper_tasks(ctx, agents, expert_response=None) 