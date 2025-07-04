from __future__ import annotations

"""Task templating utilities for ConvFinQA CrewAI multi-agent system."""

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
        return f"PRE-TEXT:\n{doc.pre_text[:400]}...\n\nTABLE:\n{table_str}"
    
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
# Task builders
# ---------------------------------------------------------------------------

def build_extraction_task(ctx: Context, agent) -> Task:
    """Create the extraction task assigned to extractor agent."""
    
    # Get available table labels for smart matching
    doc = ctx.record.doc
    available_labels = []
    if doc.table:
        all_row_labels = set()
        for column_data in doc.table.values():
            all_row_labels.update(column_data.keys())
        available_labels = list(all_row_labels)
    
    # Extract target KPIs from question
    target_kpis = ctx.extract_target_kpis()
    
    # Generate intelligent matching suggestions
    matching_suggestions = []
    for kpi in target_kpis:
        match_result = financial_matcher.find_best_match(
            query_kpi=kpi, 
            available_labels=available_labels,
            context=ctx.question
        )
        if match_result:
            matching_suggestions.append(
                f"'{kpi}' → '{match_result.matched_label}' "
                f"(confidence: {match_result.confidence:.2f}, type: {match_result.match_type})"
            )
    
    suggestion_text = "\n".join(matching_suggestions) if matching_suggestions else "No intelligent matches found"
    
    desc = f"""
CRITICAL: Extract ALL relevant numbers for this question with precise metric identification.

🎯 INTELLIGENT KPI MATCHING SUGGESTIONS:
{suggestion_text}

ENHANCED EXTRACTION STRATEGY:
1. EXACT MATCHING: Look for precise term matches first
2. SYNONYM MATCHING: Use financial domain knowledge (e.g., "repair" = "repairs and maintenance")  
3. FUZZY MATCHING: Use similarity scoring for slight variations
4. CONTEXTUAL MATCHING: Consider question context for ambiguous cases

METRIC DISAMBIGUATION RULES:
- Pay attention to EXACT wording: "net earnings" ≠ "operating income" ≠ "revenue"
- Look for MULTIPLE similar metrics and distinguish carefully
- Prioritize suggested matches above, but verify they make sense
- Search ENTIRE table, not just first few rows

FUZZY MATCHING FALLBACKS: If suggested matches don't work:
- Partial matches: "repair" matches "repairs and maintenance"
- Different formatting: "2,300" vs "2.3" (in millions)
- Similar terms: "expense" vs "costs" vs "expenditure"
- Contextual search: "decline in net earnings" → look for any earnings/income metrics
- ALWAYS find at least one number - extract ANY relevant financial metrics if specific ones missing

Return your answer as JSON exactly in this format:
    {{"candidates": [number1, number2, ...], "notes": "reasoning with EXACT metric names found and matching strategy used"}}
Do NOT call external tools. Extract numbers directly from the document text and table above.

QUESTION: "{ctx.question}"

CONVERSATION HISTORY:
{ctx.formatted_history}

DOCUMENT:
{ctx.formatted_document}
"""
    return Task(description=desc, agent=agent, expected_output="JSON string")


def build_calculation_task(ctx: Context, agent, depends_on: Task) -> Task:
    desc = f"""
Using the extracted candidates provided by coworker (task id: {depends_on.id}),
construct a minimal DSL program that answers the question.

REQUIRED DSL FORMAT - use ONLY these operations:
- For simple lookups: return the number directly (e.g., "60.94")
- For addition: "add(value1, value2)" 
- For subtraction: "subtract(value1, value2)" (value1 - value2)
- For multiplication: "multiply(value1, value2)"
- For division: "divide(value1, value2)" (value1 / value2)

CRITICAL - CONTEXT TRACKING:
- If question refers to "that", "it", "this value" → use MOST RECENT calculation result from history
- For "what is that less 1?" → subtract(previous_result, 1)
- For "what is that times 100?" → multiply(previous_result, 100)

CRITICAL - SPECIAL QUESTION PATTERNS:
- "percentage change" → "divide(subtract(new_value, old_value), old_value)"
- "ratio of X to Y" → "divide(X, Y)"
- "times 100" or "multiply by 100" → "multiply(value, 100)"
- "difference between A and B" → "subtract(A, B)" (be careful with order!)
- "change from X to Y" → "subtract(Y, X)"
- "what is that less 1" → "subtract(previous_result, 1)"

EXAMPLES:
- "What was X in 2007?" → "60.94"
- "What's the percentage change from 25.14 to 60.94?" → "divide(subtract(60.94, 25.14), 25.14)"
- "What's the ratio of 5.2 to 48.0?" → "divide(5.2, 48.0)"
- "What's that times 100?" → "multiply(2.05039, 100)" (using previous result)
- "What is that less 1?" → "subtract(2.05039, 1)" (using previous result)
- "Difference between 2008 value (93.0) and 2007 value (103.0)" → "subtract(93.0, 103.0)"

CONVERSATION CONTEXT:
{ctx.formatted_history}

CURRENT QUESTION: "{ctx.question}"

CRITICAL FORMAT RULES:
- NEVER return explanatory text, sentences, or "cannot be constructed"
- If no data found, return "0" as DSL
- If calculation impossible, return "0" as DSL
- ALWAYS return valid DSL format, never plain text

DO NOT use custom function names or explanatory text.
Return ONLY the DSL string with actual numbers.
"""
    return Task(description=desc, agent=agent, expected_output="DSL program", depends_on=[depends_on])  # type: ignore[arg-type]


def build_validation_task(ctx: Context, agent, depends_on: Task) -> Task:
    # CONSERVATIVE: Only detect explicit percentage questions to avoid over-correction
    is_percentage_question = any(word in ctx.question.lower() for word in ["percentage", "percent", "%"])
    
    scale_guidance = "📊 PERCENTAGE CONTEXT DETECTED: Question expects percentage result" if is_percentage_question else "📈 Standard numerical calculation"
    
    desc = f"""
Validate and execute the DSL program produced by coworker (task id: {depends_on.id}).

🎯 INTELLIGENT SCALE ANALYSIS:
{scale_guidance}

EXECUTION RULES:
- If DSL is a simple number (e.g., "60.94"), return it as-is
- If DSL is add(a, b), return a + b 
- If DSL is subtract(a, b), return a - b
- If DSL is multiply(a, b), return a * b  
- If DSL is divide(a, b), return a / b

SCALE-AWARE VALIDATION:
- PERCENTAGE QUESTIONS: If result is small decimal (0.296) and question asks for percentage → multiply by 100
- RATIO QUESTIONS: If result is large (>100) but question expects ratio → divide by 100  
- MAGNITUDE CHECK: Verify result magnitude makes sense for the question context
- UNIT CONSISTENCY: Check if calculation units align with question expectations

PRECISION RULES:
- Keep reasonable precision (up to 5 decimal places for percentages)
- For percentage calculations: if result is very small (< 1), keep 5 decimal places
- For large numbers: round to appropriate precision
- For ratios: keep 5 decimal places

ENHANCED EXAMPLES:
- "60.94" → 60.94
- "add(25.14, 35.8)" → 60.94
- "subtract(60.94, 25.14)" → 35.8
- "divide(subtract(60.94, 25.14), 25.14)" → 1.42403 (if percentage context: multiply by 100 → 142.403)
- "multiply(0.10826, 100)" → 10.82621

CRITICAL ERROR HANDLING:
- If DSL contains division by zero (e.g., "divide(x, 0)"), return "0" 
- If DSL is invalid or malformed, return "0"
- NEVER return "Undefined", "Cannot calculate", or explanatory text
- ALWAYS return a numerical value, even if estimated

POST-CALCULATION SCALE CHECK:
After calculating, verify the result magnitude:
- If percentage question and result < 1: consider multiplying by 100
- If ratio question and result > 1000: consider dividing by 100
- If scale mismatch obvious: apply appropriate correction

Return ONLY the final numerical result with appropriate scale adjustments.
Do NOT return the DSL program itself or any explanatory text.
"""
    return Task(description=desc, agent=agent, expected_output="numerical answer", depends_on=[depends_on])  # type: ignore[arg-type]


def build_tasks(ctx: Context, agents: dict[str, Any]) -> List[Task]:
    """Create the canonical extractor → calculator → validator task chain."""
    extract_task = build_extraction_task(ctx, agents["extractor"])
    calc_task = build_calculation_task(ctx, agents["calculator"], extract_task)
    validate_task = build_validation_task(ctx, agents["validator"], calc_task)
    return [extract_task, calc_task, validate_task] 