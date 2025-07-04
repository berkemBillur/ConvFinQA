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


# ---------------------------------------------------------------------------
# Six-Agent Task Builders
# ---------------------------------------------------------------------------

def build_manager_task(ctx: Context, agent, conversation_cache: dict | None = None) -> Task:
    """Create conversation manager task for routing and cache decisions."""
    
    cache_info = ""
    if conversation_cache:
        cache_entries = []
        for key, value in conversation_cache.items():
            cache_entries.append(f"Q: {key} | A: {value}")
        cache_info = f"CACHE CONTENTS:\n" + "\n".join(cache_entries[-5:])  # Last 5 entries
    else:
        cache_info = "CACHE: Empty"
    
    desc = f"""
CONVERSATION MANAGER TASK - Route question efficiently

CURRENT QUESTION: "{ctx.question}"

CONVERSATION HISTORY:
{ctx.formatted_history}

{cache_info}

ROUTING DECISION:
Analyze the current question and decide:
1. Can this be answered from cache? (exact match or trivial variation)
2. Or does it need the full pipeline?

Common cache-hit patterns:
- "Same as above" / "Same as before" → use most recent answer
- "What about [previous year]?" → if we calculated a similar metric recently
- Exact question repeats

IMPORTANT: Only use cache for EXACT matches or TRIVIAL variations. When in doubt, use full pipeline.

Return JSON in this exact format:
{{
  "action": "cache_hit" | "run_pipeline",
  "cached_answer": "answer_if_cache_hit_or_empty_string",
  "reasoning": "brief explanation of decision"
}}
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"action": "run_pipeline", "cached_answer": "", "reasoning": "New question requires full analysis"}'
    )


def build_extractor_task(ctx: Context, agent) -> Task:
    """Create data extraction specialist task."""
    
    iteration_context = ""
    if ctx.iteration > 0:
        iteration_context = f"""
REVISION ITERATION {ctx.iteration}:
You are revising your previous extraction based on critic feedback.
Critic feedback to address:
{ctx.critic_feedback}

Please improve your data extraction by addressing the specific issues raised.
Focus on any extraction accuracy, completeness, or relevance problems mentioned.
"""
    
    desc = f"""
DATA EXTRACTION SPECIALIST TASK - Locate and extract exact financial data

{iteration_context}

QUESTION: "{ctx.question}"

DOCUMENT CONTEXT: {ctx.formatted_document}

EXTRACTION STRATEGY:

1. UNDERSTAND THE REQUEST:
   - What specific metrics/values are needed?
   - What time periods are involved?
   - Are there conversational references to resolve?

2. SYSTEMATIC TABLE SEARCH:
   - Scan ALL rows and columns systematically
   - Look for EXACT metric matches first, then synonyms
   - Pay attention to time period columns vs value columns
   - Don't confuse year labels (2020, 2021) with actual financial values

3. VALUE NORMALIZATION:
   - Extract the RAW value exactly as shown
   - Note any unit indicators ("in thousands", "in millions")
   - Convert parentheses to negative signs: (1,234) → -1234
   - Handle scale multipliers correctly

4. REFERENCE RESOLUTION:
   - "That value" → use most recent result from conversation history
   - "Previous year" → identify specific year from context
   - "The two values" → extract both values mentioned

5. QUALITY VERIFICATION:
   - Do the extracted values actually answer the question?
   - Are all necessary data points included?
   - Are row/column references accurate?
   - Is the time period selection correct?

Return JSON in this EXACT format:
{{
  "extractions": [
    {{
      "row": "Total Revenue",
      "col": "2022",
      "raw": "60,940",
      "unit": "millions",
      "scale": 1000000
    }},
    {{
      "row": "Total Revenue", 
      "col": "2021",
      "raw": "45,230",
      "unit": "millions",
      "scale": 1000000
    }}
  ],
  "references_resolved": ["Used 2022 and 2021 for year-over-year comparison"],
  "extraction_notes": "Extracted total revenue for both years to calculate growth rate"
}}

CRITICAL:
- Extract ALL values needed to answer the question
- Be precise about row and column references
- Handle units and scaling correctly
- Resolve any conversational references
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"extractions": [], "references_resolved": [], "extraction_notes": ""}'
    )


def build_reasoner_task(ctx: Context, agent, extractor_output: str = "") -> Task:
    """Create fin-calc reasoner task."""
    
    iteration_context = ""
    if ctx.iteration > 0:
        iteration_context = f"""
REVISION ITERATION {ctx.iteration}:
You are revising your previous calculation based on critic feedback.
Critic feedback to address:
{ctx.critic_feedback}

Please improve your calculation by addressing the specific issues raised.
"""
    
    extraction_data_section = ""
    try:
        import json as _json
        _eo = _json.loads(extractor_output) if extractor_output else {}
        if isinstance(_eo, dict) and _eo.get("extractions"):
            lines = []
            for item in _eo["extractions"]:
                raw_str = str(item.get("raw"))
                unit = str(item.get("unit"))
                scale = item.get("scale", 1)
                try:
                    # Remove commas/parentheses for numeric parse
                    numeric_raw = float(str(raw_str).replace(",", "").replace("(", "-").replace(")", ""))
                    scaled_val = numeric_raw * float(scale)
                    scaled_disp = f"{scaled_val:.6g}"
                except Exception:
                    scaled_disp = "non-numeric"
                lines.append(f"- {item.get('row')} | {item.get('col')}: {raw_str} × {scale} → {scaled_disp}")
            if lines:
                extraction_data_section = "\nSCALED VALUES (auto-parsed):\n" + "\n".join(lines) + "\n"
    except Exception:
        pass

    desc = f"""
FINANCIAL CALCULATION REASONER TASK - Perform precise calculations

{iteration_context}

QUESTION: "{ctx.question}"

{extraction_data_section}

EXTRACTED DATA: {extractor_output}

CALCULATION STRATEGY:

1. UNDERSTAND THE CALCULATION TYPE:
   - Simple lookup (extract single value)
   - Percentage change: (new - old) / old * 100
   - Ratio: numerator / denominator  
   - Difference: value1 - value2
   - Growth rate: (current - previous) / previous

2. APPLY EXTRACTED VALUES:
   - Use the exact values from the extractor JSON
   - Apply scale multipliers correctly
   - Preserve signs and handle negatives properly

3. STEP-BY-STEP REASONING:
   - Show each calculation step clearly
   - Explain the formula being used
   - Verify the calculation makes business sense

4. DSL GENERATION (when needed):
   - For complex calculations, generate executable DSL
   - Use: add(), subtract(), multiply(), divide()
   - Example: "divide(60.94, 45.23)" for ratio calculations

5. CALCULATOR TOOL USAGE:
   - Use the calculator tool for complex arithmetic
   - Especially when dealing with multiple operations
   - Ensures precision and reduces calculation errors

Return JSON in this EXACT format:
{{
  "steps": [
    "Step 1: Extract revenue for 2022: 60.94 million (scale: 1e6) = 60,940,000",
    "Step 2: Extract revenue for 2021: 45.23 million (scale: 1e6) = 45,230,000", 
    "Step 3: Calculate growth rate: (60,940,000 - 45,230,000) / 45,230,000 = 0.347",
    "Step 4: Convert to percentage: 0.347 * 100 = 34.7%"
  ],
  "dsl": "divide(subtract(60940000, 45230000), 45230000)",
  "answer": "0.347"
}}

CRITICAL:
- Show your work step by step
- Use exact values from extractor JSON
- Generate DSL for verification when appropriate
- Final answer should be the numeric result only
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"steps": [], "dsl": "", "answer": ""}'
    )


def build_extraction_critic_task_six(ctx: Context, extractor_output: str, agent) -> Task:
    """Create extraction critic task for six-agent workflow."""
    
    desc = f"""
EXTRACTION CRITIC TASK - Review data extraction quality

ORIGINAL QUESTION: "{ctx.question}"
DOCUMENT CONTEXT: {ctx.formatted_document}
EXTRACTOR OUTPUT TO REVIEW: {extractor_output}

EVALUATION CRITERIA:
1. RELEVANCE: Are the extracted values actually relevant to the question?
2. COMPLETENESS: Were any important data points missed?
3. ACCURACY: Are the row/column references correct?
4. SCALING: Are units and scale multipliers handled properly?
5. TEMPORAL: Are the correct time periods/years selected?
6. REFERENCES: Are conversational references properly resolved?

SPECIFIC CHECKS:
- Verify each extraction against the document table
- Check that row and column labels match exactly
- Ensure scale multipliers are correct (thousands=1000, millions=1000000)
- Verify negative values are handled properly
- Confirm all values needed for the calculation are extracted
- If the `extractions` array is empty, or any `raw` field equals "N/A", "NA", "Data not available", or is non-numeric, you MUST set `is_correct` to `false`.

Provide your assessment in EXACTLY this JSON format:
{{
  "is_correct": <true | false>,
  "issues": ["issue 1", "issue 2"],
  "suggested_fix": "concise suggestion if false, empty string if true"
}}

Guidelines:
- is_correct: true only when extraction is fully accurate and complete
- When false, list specific concrete issues
- Suggest specific fixes (e.g., "Include 2021 data from row X")
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"is_correct": true, "issues": [], "suggested_fix": ""}'
    )


def build_calculation_critic_task_six(ctx: Context, reasoner_output: str, agent) -> Task:
    """Create calculation critic task for six-agent workflow."""
    
    desc = f"""
CALCULATION CRITIC TASK - Review mathematical reasoning

ORIGINAL QUESTION: "{ctx.question}"
REASONER OUTPUT TO REVIEW: {reasoner_output}

EVALUATION CRITERIA:
1. METHOD: Is the calculation method appropriate for the question type?
2. OPERATIONS: Are mathematical operations performed correctly?
3. LOGIC: Is the step-by-step reasoning sound?
4. SCALE: Are scale multipliers applied correctly?
5. BUSINESS_SENSE: Does the result magnitude make sense?
6. DSL: Is the generated DSL program correct?

SPECIFIC CHECKS:
- Verify each calculation step is mathematically correct
- Check that formulas match the question type (growth, ratio, etc.)
- Ensure scale conversions are applied properly
- Verify the final answer makes business sense
- Check DSL syntax and logic if provided
- If a `dsl` program is present, mentally (or using rough calculation) replay it and compare the result against the `answer` field.  If they differ, set `is_correct` to `false`.

Common calculation patterns to verify:
- Percentage change: (new - old) / old * 100
- Growth rates: (current - previous) / previous
- Ratios: numerator / denominator
- Differences: value1 - value2

Provide your assessment in EXACTLY this JSON format:
{{
  "is_correct": <true | false>,
  "issues": ["issue 1", "issue 2"],
  "suggested_fix": "concise suggestion if false, empty string if true"
}}

Guidelines:
- is_correct: true only when calculation logic is entirely correct
- When false, enumerate specific mathematical problems
- Suggest specific fixes (e.g., "Use (new-old)/old formula for percentage change")
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"is_correct": true, "issues": [], "suggested_fix": ""}'
    )


def build_synthesiser_task(ctx: Context, reasoner_output: str, extraction_critic_output: str, calculation_critic_output: str, agent) -> Task:
    """Create answer synthesiser task."""
    
    desc = f"""
ANSWER SYNTHESISER TASK - Create final response

ORIGINAL QUESTION: "{ctx.question}"
REASONER OUTPUT: {reasoner_output}
EXTRACTION CRITIC: {extraction_critic_output}
CALCULATION CRITIC: {calculation_critic_output}

SYNTHESIS STRATEGY:

1. CRITIC ANALYSIS:
   - Parse both critic responses for is_correct flags
   - If BOTH critics say is_correct: true → proceed to final answer
   - If ANY critic says is_correct: false → request revision

2. FINAL ANSWER FORMATTING (when approved):
   - Extract the numeric answer from reasoner output
   - Apply any needed post-processing (scale normalization, etc.)
   - Format in conversational style appropriate for financial Q&A
   - Do NOT include step-by-step reasoning in final answer

3. REVISION REQUEST (when not approved):
   - Synthesize critic feedback into clear, actionable guidance
   - Highlight the most critical issues to address
   - Provide specific suggestions for improvement

Return JSON in this EXACT format:

For FINAL answers (both critics approve):
{{
  "status": "final",
  "answer": "34.7%",
  "critique_summary": "Both critics approved the extraction and calculation"
}}

For REVISION requests (critics found issues):
{{
  "status": "revise", 
  "answer": "",
  "critique_summary": "Extraction critic: missing 2021 data. Calculation critic: incorrect formula used."
}}

CRITICAL:
- Only return status: "final" when BOTH critics say is_correct: true
- Final answers should be conversational but precise
- Revision feedback should be specific and actionable
"""

    return Task(
        description=desc,
        agent=agent,
        expected_output='{"status": "final", "answer": "", "critique_summary": ""}'
    )


def build_six_agent_tasks(ctx: Context, agents: dict[str, Any], 
                         stage: str = "manager",
                         extractor_output: str = "",
                         reasoner_output: str = "",
                         extraction_critic_output: str = "",
                         calculation_critic_output: str = "",
                         conversation_cache: dict | None = None) -> List[Task]:
    """Build tasks for the six-agent workflow based on current stage.
    
    Args:
        ctx: Context object with question and document info
        agents: Dictionary of agent objects
        stage: Current workflow stage ("manager", "extraction", "reasoning", "critics", "synthesis")
        extractor_output: Output from extractor (for later stages)
        reasoner_output: Output from reasoner (for critics and synthesis)
        extraction_critic_output: Output from extraction critic (for synthesis)
        calculation_critic_output: Output from calculation critic (for synthesis)
        conversation_cache: Cache for manager decisions
    
    Returns:
        List of tasks for the current stage
    """
    if stage == "manager":
        return [build_manager_task(ctx, agents["manager"], conversation_cache)]
    
    elif stage == "extraction":
        return [build_extractor_task(ctx, agents["extractor"])]
    
    elif stage == "reasoning":
        return [build_reasoner_task(ctx, agents["reasoner"], extractor_output)]
    
    elif stage == "critics":
        return [
            build_extraction_critic_task_six(ctx, extractor_output, agents["extraction_critic"]),
            build_calculation_critic_task_six(ctx, reasoner_output, agents["calculation_critic"])
        ]
    
    elif stage == "synthesis":
        return [build_synthesiser_task(ctx, reasoner_output, extraction_critic_output, 
                                     calculation_critic_output, agents["synthesiser"])]
    
    else:
        raise ValueError(f"Unknown stage: {stage}") 