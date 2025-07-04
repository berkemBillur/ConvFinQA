"""Utility functions for ConvFinQA data processing."""

from typing import Dict, List, Tuple, Union, Any, Optional, NamedTuple
import pandas as pd
import re
from dataclasses import dataclass

from .models import ConvFinQARecord, Document, Dialogue


@dataclass
class ValueCandidate:
    """Represents a candidate numerical value for DSL generation."""
    value: float
    source_text: str
    context: str
    score: float
    location: Dict[str, Any]  # Table position, header info, etc.


@dataclass 
class ScoringWeights:
    """Weights for value extraction scoring."""
    header_match: float = 0.4
    keyword_proximity: float = 0.3
    question_alignment: float = 0.2
    financial_relevance: float = 0.1


def extract_table_data(document: Document) -> pd.DataFrame:
    """Convert document table to pandas DataFrame.
    
    Args:
        document: ConvFinQA document containing table data.
        
    Returns:
        DataFrame representation of the table.
    """
    if not document.table:
        return pd.DataFrame()
    
    # Convert the nested dict structure to DataFrame
    return pd.DataFrame(document.table)


def get_table_column_names(document: Document) -> List[str]:
    """Get column names from document table.
    
    Args:
        document: ConvFinQA document.
        
    Returns:
        List of column names.
    """
    if not document.table:
        return []
    return list(document.table.keys())


def get_table_row_names(document: Document) -> List[str]:
    """Get row names from document table.
    
    Args:
        document: ConvFinQA document.
        
    Returns:
        List of row names.
    """
    if not document.table:
        return []
    
    # Get row names from first column
    first_column = next(iter(document.table.values()))
    return list(first_column.keys())


def extract_numeric_values(table: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Extract numeric values from table, converting where possible.
    
    Args:
        table: Raw table data.
        
    Returns:
        Table with numeric values extracted.
    """
    numeric_table = {}
    
    for col_name, col_data in table.items():
        numeric_col = {}
        for row_name, value in col_data.items():
            if isinstance(value, (int, float)):
                numeric_col[row_name] = float(value)
            elif isinstance(value, str):
                # Try to convert string to number
                try:
                    # Handle common financial notation
                    cleaned_value = value.replace(',', '').replace('$', '').strip()
                    
                    # Handle parentheses for negative numbers
                    if cleaned_value.startswith('(') and cleaned_value.endswith(')'):
                        cleaned_value = '-' + cleaned_value[1:-1]
                    
                    numeric_col[row_name] = float(cleaned_value)
                except ValueError:
                    # Keep original string if can't convert
                    numeric_col[row_name] = value
            else:
                numeric_col[row_name] = value
                
        numeric_table[col_name] = numeric_col
    
    return numeric_table


def get_conversation_context(dialogue: Dialogue, turn_index: int) -> str:
    """Build conversation context up to a specific turn.
    
    Args:
        dialogue: ConvFinQA dialogue.
        turn_index: Index of the current turn (0-based).
        
    Returns:
        Formatted conversation context.
    """
    context_parts = []
    
    for i in range(min(turn_index, len(dialogue.conv_questions))):
        question = dialogue.conv_questions[i]
        answer = dialogue.conv_answers[i] if i < len(dialogue.conv_answers) else "No answer"
        
        context_parts.append(f"Q{i+1}: {question}")
        context_parts.append(f"A{i+1}: {answer}")
    
    return "\n".join(context_parts)


def format_table_for_prompt(document: Document, max_rows: int = 10) -> str:
    """Format table data for inclusion in prompts.
    
    Args:
        document: ConvFinQA document.
        max_rows: Maximum number of rows to include.
        
    Returns:
        Formatted table string.
    """
    if not document.table:
        return "No table data available."
    
    df = extract_table_data(document)
    
    if df.empty:
        return "Empty table."
    
    # Limit rows if too many
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated_note = f"\n... (showing first {max_rows} rows)"
    else:
        truncated_note = ""
    
    return df.to_string() + truncated_note


def get_turn_dependencies(dialogue: Dialogue) -> List[List[int]]:
    """Analyse dependencies between conversation turns.
    
    Args:
        dialogue: ConvFinQA dialogue.
        
    Returns:
        List where each element contains indices of turns this turn depends on.
    """
    dependencies = []
    
    for i, question in enumerate(dialogue.conv_questions):
        turn_deps = []
        
        # Simple heuristic: look for references to previous answers
        question_lower = question.lower()
        
        # Check for pronouns and relative references
        if any(word in question_lower for word in ['that', 'this', 'it', 'what about']):
            # Likely depends on previous turn
            if i > 0:
                turn_deps.append(i - 1)
        
        # Check for year references that might connect to previous context
        if any(word in question_lower for word in ['same year', 'previous year', 'next year']):
            # Find recent turns mentioning years
            for j in range(max(0, i - 3), i):
                if any(char.isdigit() for char in dialogue.conv_questions[j]):
                    turn_deps.append(j)
        
        dependencies.append(turn_deps)
    
    return dependencies


def validate_record(record: ConvFinQARecord) -> List[str]:
    """Validate a ConvFinQA record for common issues.
    
    Args:
        record: ConvFinQA record to validate.
        
    Returns:
        List of validation warnings/errors.
    """
    issues = []
    
    # Check dialogue consistency
    dialogue = record.dialogue
    n_questions = len(dialogue.conv_questions)
    n_answers = len(dialogue.conv_answers)
    n_programs = len(dialogue.turn_program)
    n_executed = len(dialogue.executed_answers)
    
    if not all(x == n_questions for x in [n_answers, n_programs, n_executed]):
        issues.append(f"Dialogue length mismatch: {n_questions} questions, {n_answers} answers, {n_programs} programs, {n_executed} executed")
    
    # Check features consistency
    if record.features.num_dialogue_turns != n_questions:
        issues.append(f"Features mismatch: {record.features.num_dialogue_turns} != {n_questions}")
    
    # Check table structure
    if record.doc.table:
        col_lengths = [len(col_data) for col_data in record.doc.table.values()]
        if len(set(col_lengths)) > 1:
            issues.append(f"Table columns have different lengths: {col_lengths}")
    
    return issues


def extract_value_candidates(
    document: Document, 
    question: str, 
    financial_keywords: Optional[List[str]] = None,
    max_candidates: int = 5
) -> List[ValueCandidate]:
    """Extract and score potential numerical values for DSL generation.
    
    Args:
        document: Document containing table and text.
        question: Question text for context matching.
        financial_keywords: Keywords indicating financial relevance.
        max_candidates: Maximum number of candidates to return.
        
    Returns:
        List of scored value candidates, sorted by score descending.
    """
    if financial_keywords is None:
        financial_keywords = ["revenue", "profit", "income", "expense", "cost", "sales"]
    
    candidates = []
    question_lower = question.lower()
    
    # Extract from table
    if document.table:
        numeric_table = extract_numeric_values(document.table)
        candidates.extend(_extract_table_candidates(
            numeric_table, question_lower, financial_keywords
        ))
    
    # Extract numbers mentioned in question
    candidates.extend(_extract_question_candidates(question))
    
    # Score and rank candidates
    scored_candidates = []
    for candidate in candidates:
        score = _calculate_candidate_score(candidate, question_lower, financial_keywords)
        candidate.score = score
        scored_candidates.append(candidate)
    
    # Sort by score and return top candidates
    scored_candidates.sort(key=lambda x: x.score, reverse=True)
    return scored_candidates[:max_candidates]


def _extract_table_candidates(
    numeric_table: Dict[str, Dict[str, Any]], 
    question_lower: str,
    financial_keywords: List[str]
) -> List[ValueCandidate]:
    """Extract value candidates from table data."""
    candidates = []
    
    for col_name, col_data in numeric_table.items():
        for row_name, value in col_data.items():
            if isinstance(value, (int, float)):
                candidate = ValueCandidate(
                    value=float(value),
                    source_text=str(value),
                    context=f"{col_name} - {row_name}",
                    score=0.0,  # Will be calculated later
                    location={
                        "column": col_name,
                        "row": row_name,
                        "type": "table"
                    }
                )
                candidates.append(candidate)
    
    return candidates


def _extract_question_candidates(question: str) -> List[ValueCandidate]:
    """Extract numerical values mentioned in the question."""
    candidates = []
    
    # Find numbers in question text
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.finditer(number_pattern, question)
    
    for match in matches:
        try:
            # Clean and convert number
            number_str = match.group().replace(',', '')
            value = float(number_str)
            
            candidate = ValueCandidate(
                value=value,
                source_text=match.group(),
                context=f"mentioned in question",
                score=0.0,
                location={
                    "start": match.start(),
                    "end": match.end(),
                    "type": "question"
                }
            )
            candidates.append(candidate)
        except ValueError:
            continue
    
    return candidates


def _calculate_candidate_score(
    candidate: ValueCandidate,
    question_lower: str, 
    financial_keywords: List[str],
    weights: Optional[ScoringWeights] = None
) -> float:
    """Calculate relevance score for a value candidate."""
    if weights is None:
        weights = ScoringWeights()
    
    score = 0.0
    
    # Header/context matching score
    context_lower = candidate.context.lower()
    context_score = 0.0
    
    # Check for financial keyword matches
    for keyword in financial_keywords:
        if keyword in context_lower:
            context_score += 0.5
    
    # Check for question keyword matches
    question_words = set(question_lower.split())
    context_words = set(context_lower.split())
    keyword_overlap = len(question_words.intersection(context_words))
    context_score += keyword_overlap * 0.1
    
    score += context_score * weights.header_match
    
    # Question alignment score (if number appears in question)
    if candidate.location.get("type") == "question":
        score += 1.0 * weights.question_alignment
    
    # Financial relevance score
    financial_score = 0.0
    for keyword in financial_keywords:
        if keyword in question_lower:
            financial_score += 0.2
    
    score += financial_score * weights.financial_relevance
    
    # Keyword proximity score (simple heuristic)
    proximity_score = 0.0
    if any(word in context_lower for word in question_lower.split()):
        proximity_score = 0.5
    
    score += proximity_score * weights.keyword_proximity
    
    return score


def detect_financial_patterns(document: Document) -> Dict[str, Any]:
    """Detect common financial patterns in table data.
    
    Args:
        document: Document to analyse.
        
    Returns:
        Dictionary of detected patterns and their locations.
    """
    patterns = {
        "years": [],
        "currency_columns": [],
        "percentage_values": [],
        "large_numbers": []
    }
    
    if not document.table:
        return patterns
    
    # Detect year patterns
    for col_name, col_data in document.table.items():
        # Check column headers for years
        year_match = re.search(r'\b(19|20)\d{2}\b', col_name)
        if year_match:
            patterns["years"].append({
                "column": col_name,
                "year": int(year_match.group()),
                "type": "header"
            })
        
        # Check for currency indicators
        if any(indicator in col_name.lower() for indicator in ['$', 'revenue', 'income', 'cost']):
            patterns["currency_columns"].append(col_name)
        
        # Check for percentage patterns in values
        for row_name, value in col_data.items():
            if isinstance(value, str) and '%' in value:
                patterns["percentage_values"].append({
                    "column": col_name,
                    "row": row_name,
                    "value": value
                })
            
            # Detect large numbers (potential financial figures)
            if isinstance(value, (int, float)) and abs(value) > 1000:
                patterns["large_numbers"].append({
                    "column": col_name,
                    "row": row_name,
                    "value": value
                })
    
    return patterns  