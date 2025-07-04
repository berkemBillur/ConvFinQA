"""Pydantic schema models for six-agent workflow outputs.

This module defines strict JSON schemas for each agent's expected output,
enabling validation and error handling in the orchestrator.
"""

from typing import List, Literal, Optional, Union, Dict, Any, Type, TypeVar
from pydantic import BaseModel, Field, validator
import logging
import json
import re
from enum import Enum

logger = logging.getLogger(__name__)

# Generic type variable for schema fallback methods
T = TypeVar('T', bound=BaseModel)


class ValidationError(Exception):
    """Custom exception for schema validation errors."""
    def __init__(self, message: str, raw_output: str = "", retry_count: int = 0):
        self.message = message
        self.raw_output = raw_output
        self.retry_count = retry_count
        super().__init__(message)


class ValidationStrategy(Enum):
    """Strategy for handling validation failures."""
    STRICT = "strict"  # Raise exception, no fallback
    GRACEFUL = "graceful"  # Return safe fallback
    RETRY = "retry"  # Attempt to fix and retry
    

class ExtractionItem(BaseModel):
    """Individual extraction item from the data extraction specialist."""
    row: str = Field(..., description="Row label from the financial table")
    col: str = Field(..., description="Column label from the financial table")
    raw: str = Field(..., description="Raw extracted value as string")
    unit: str = Field(..., description="Unit description (e.g., 'million', 'thousand')")
    scale: float = Field(..., description="Scale multiplier (e.g., 1000000 for millions)")

    @validator('scale')
    def validate_scale(cls, v):
        """Ensure scale is a positive number."""
        if v <= 0:
            raise ValueError("Scale must be positive")
        return v


class ManagerOutput(BaseModel):
    """Output schema for Conversation Manager agent."""
    action: Literal["cache_hit", "run_pipeline"] = Field(
        ..., description="Decision on whether to use cache or run full pipeline"
    )
    cached_answer: str = Field(
        ..., description="Cached answer if action is cache_hit, empty string otherwise"
    )
    reasoning: str = Field(
        ..., description="Brief explanation of the routing decision"
    )

    @validator('cached_answer')
    def validate_cached_answer(cls, v, values):
        """Ensure cached_answer is provided when action is cache_hit."""
        if values.get('action') == 'cache_hit' and not v.strip():
            raise ValueError("cached_answer must be provided when action is cache_hit")
        return v


class ExtractorOutput(BaseModel):
    """Output schema for Data Extraction Specialist agent."""
    extractions: List[ExtractionItem] = Field(
        ..., description="List of extracted data items from the financial document"
    )
    references_resolved: List[str] = Field(
        default_factory=list, description="Explanations of resolved conversational references"
    )
    extraction_notes: str = Field(
        default="", description="Additional notes about the extraction process"
    )

    @validator('extractions')
    def validate_extractions_not_empty(cls, v):
        """Ensure at least one extraction is provided."""
        if not v:
            raise ValueError("At least one extraction must be provided")
        return v


class ReasonerOutput(BaseModel):
    """Output schema for Financial Calculation Reasoner agent."""
    steps: List[str] = Field(
        ..., description="Step-by-step reasoning chain for the calculation"
    )
    dsl: Optional[str] = Field(
        default="", description="Optional DSL program for complex calculations"
    )
    answer: str = Field(
        ..., description="Final numeric answer as string"
    )

    @validator('steps')
    def validate_steps_not_empty(cls, v):
        """Ensure at least one reasoning step is provided."""
        if not v:
            raise ValueError("At least one reasoning step must be provided")
        return v

    @validator('answer')
    def validate_answer_not_empty(cls, v):
        """Ensure answer is not empty."""
        if not v.strip():
            raise ValueError("Answer cannot be empty")
        return v


class CriticOutput(BaseModel):
    """Output schema for both Extraction and Calculation Critic agents."""
    is_correct: bool = Field(
        ..., description="Whether the reviewed work is correct"
    )
    issues: List[str] = Field(
        default_factory=list, description="List of identified issues"
    )
    suggested_fix: str = Field(
        default="", description="Suggested fix if is_correct is false, empty if correct"
    )

    @validator('issues')
    def validate_issues_when_incorrect(cls, v, values):
        """Ensure issues are provided when is_correct is false."""
        if not values.get('is_correct', True) and not v:
            raise ValueError("Issues must be provided when is_correct is false")
        return v


class SynthesiserOutput(BaseModel):
    """Output schema for Answer Synthesiser agent."""
    status: Literal["final", "revise"] = Field(
        ..., description="Whether to provide final answer or request revision"
    )
    answer: str = Field(
        ..., description="Final answer when status is final, empty when revising"
    )
    critique_summary: str = Field(
        default="", description="Summary of critic feedback"
    )

    @validator('answer')
    def validate_answer_when_final(cls, v, values):
        """Ensure answer is provided when status is final."""
        if values.get('status') == 'final' and not v.strip():
            raise ValueError("Answer must be provided when status is final")
        return v


class RobustSchemaValidator:
    """Enhanced schema validator with proper error recovery and graceful degradation."""
    
    def __init__(self, max_retries: int = 2, enable_fallbacks: bool = True):
        self.max_retries = max_retries
        self.enable_fallbacks = enable_fallbacks
        self._validation_stats = {}
    
    def validate_with_recovery(
        self, 
        raw_output: str, 
        schema_class: Type[BaseModel], 
        agent_name: str,
        strategy: ValidationStrategy = ValidationStrategy.GRACEFUL
    ) -> BaseModel:
        """
        Validate agent output with comprehensive error recovery.
        
        Args:
            raw_output: Raw agent output string
            schema_class: Pydantic model class to validate against
            agent_name: Name of the agent for logging
            strategy: Validation strategy to use
        
        Returns:
            Validated schema instance
            
        Raises:
            ValidationError: When strict validation fails
        """
        # Track validation attempts
        if agent_name not in self._validation_stats:
            self._validation_stats[agent_name] = {"attempts": 0, "failures": 0, "fallbacks": 0}
        
        self._validation_stats[agent_name]["attempts"] += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                # Extract and validate JSON
                json_data = self._extract_json_safely(raw_output, agent_name)
                if not json_data:
                    raise ValidationError(f"No valid JSON found in {agent_name} output", raw_output, attempt)
                
                # Attempt schema validation
                validated = schema_class(**json_data)
                
                # Log success
                if attempt > 0:
                    logger.info(f"✅ {agent_name} validation succeeded on attempt {attempt + 1}")
                
                return validated
                
            except Exception as e:
                logger.warning(f"🔄 {agent_name} validation attempt {attempt + 1} failed: {str(e)[:100]}...")
                
                if attempt < self.max_retries:
                    # Try to clean and fix the JSON for next attempt
                    raw_output = self._attempt_json_repair(raw_output, str(e))
                    continue
                
                # All retries exhausted
                self._validation_stats[agent_name]["failures"] += 1
                
                if strategy == ValidationStrategy.STRICT:
                    raise ValidationError(f"{agent_name} validation failed after {self.max_retries + 1} attempts: {e}", raw_output, attempt)
                
                elif strategy == ValidationStrategy.GRACEFUL and self.enable_fallbacks:
                    self._validation_stats[agent_name]["fallbacks"] += 1
                    fallback_result = self._create_safe_fallback(schema_class, agent_name, str(e))
                    # Type cast is safe here since _create_safe_fallback returns the correct type
                    return fallback_result  # type: ignore
                
                else:
                    raise ValidationError(f"{agent_name} validation failed: {e}", raw_output, attempt)
    
    def _extract_json_safely(self, raw_output: str, agent_name: str) -> Dict[str, Any]:
        """Safely extract JSON from raw agent output."""
        if not raw_output or not raw_output.strip():
            raise ValueError("Empty output")
        
        # Try multiple JSON extraction strategies
        strategies = [
            # Strategy 1: Find complete JSON object
            lambda text: self._find_complete_json(text),
            # Strategy 2: Find JSON by braces
            lambda text: self._find_json_by_braces(text),
            # Strategy 3: Extract from code blocks
            lambda text: self._find_json_in_code_blocks(text),
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                json_data = strategy(raw_output)
                if json_data:
                    logger.debug(f"✅ {agent_name} JSON extracted using strategy {i + 1}")
                    return json_data
            except Exception as e:
                logger.debug(f"❌ {agent_name} JSON extraction strategy {i + 1} failed: {e}")
                continue
        
        raise ValueError("No valid JSON found using any extraction strategy")
    
    def _find_complete_json(self, text: str) -> Dict[str, Any]:
        """Find complete JSON object in text."""
        # Look for JSON object with proper nesting
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_str = text[start_pos:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
        
        raise ValueError("No complete JSON object found")
    
    def _find_json_by_braces(self, text: str) -> Dict[str, Any]:
        """Find JSON using regex pattern."""
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        raise ValueError("No valid JSON found by brace matching")
    
    def _find_json_in_code_blocks(self, text: str) -> Dict[str, Any]:
        """Find JSON in markdown code blocks."""
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?})`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        raise ValueError("No valid JSON found in code blocks")
    
    def _attempt_json_repair(self, raw_output: str, error_message: str) -> str:
        """Attempt to repair common JSON issues."""
        repaired = raw_output
        
        # Common fixes
        repairs = [
            # Fix unescaped quotes
            (r'(?<!\\)"([^"]*)"([^,}\]]*)"', r'"\1\2"'),
            # Fix trailing commas
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix missing quotes on keys
            (r'(\w+):', r'"\1":'),
        ]
        
        for pattern, replacement in repairs:
            try:
                repaired = re.sub(pattern, replacement, repaired)
            except Exception:
                continue
        
        return repaired
    
    def _create_safe_fallback(self, schema_class: Type[T], agent_name: str, error: str) -> T:
        """Create safe fallback instances that won't break the workflow."""
        logger.error(f"🚨 Creating fallback for {agent_name} due to: {error}")
        
        if schema_class == ManagerOutput:
            # Safe fallback: proceed with pipeline, don't use cache
            return ManagerOutput(
                action="run_pipeline",
                cached_answer="",
                reasoning=f"Validation fallback: proceeding with full pipeline due to parsing error"
            )
        
        elif schema_class == ExtractorOutput:
            # Safe fallback: indicate extraction failed, but don't break pipeline
            return ExtractorOutput(
                extractions=[ExtractionItem(
                    row="VALIDATION_ERROR", 
                    col="UNKNOWN", 
                    raw="0", 
                    unit="", 
                    scale=1.0
                )],
                references_resolved=[],
                extraction_notes=f"⚠️ Extraction validation failed: {error[:100]}. Using fallback."
            )
        
        elif schema_class == ReasonerOutput:
            # Safe fallback: provide minimal reasoning, indicate error
            return ReasonerOutput(
                steps=[f"⚠️ Reasoning validation failed: {error[:100]}. Using fallback calculation."],
                dsl="",
                answer="0"
            )
        
        elif schema_class == CriticOutput:
            # Safe fallback: Don't approve (crucial for quality control)
            return CriticOutput(
                is_correct=False,
                issues=[f"Validation failed: {error[:100]}"],
                suggested_fix="Agent output validation failed. Please retry with properly formatted JSON."
            )
        
        elif schema_class == SynthesiserOutput:
            # Safe fallback: Request revision instead of forcing final answer
            return SynthesiserOutput(
                status="revise",
                answer="",
                critique_summary=f"Synthesis validation failed: {error[:100]}. Requesting revision."
            )
        
        else:
            raise ValidationError(f"No safe fallback available for {schema_class.__name__}")
    
    def get_validation_stats(self) -> Dict[str, Dict[str, int]]:
        """Get validation statistics for monitoring."""
        return self._validation_stats.copy()


# Enhanced validator instance - maintains backward compatibility
class SchemaValidator:
    """Backward compatible wrapper for enhanced validation."""
    
    _validator = RobustSchemaValidator(max_retries=2, enable_fallbacks=True)
    
    @staticmethod
    def validate_manager_output(raw_output: str) -> ManagerOutput:
        """Validate and parse manager agent output."""
        return SchemaValidator._validator.validate_with_recovery(
            raw_output, ManagerOutput, "manager", ValidationStrategy.GRACEFUL
        )

    @staticmethod
    def validate_extractor_output(raw_output: str) -> ExtractorOutput:
        """Validate and parse extractor agent output."""
        return SchemaValidator._validator.validate_with_recovery(
            raw_output, ExtractorOutput, "extractor", ValidationStrategy.GRACEFUL
        )

    @staticmethod
    def validate_reasoner_output(raw_output: str) -> ReasonerOutput:
        """Validate and parse reasoner agent output."""
        return SchemaValidator._validator.validate_with_recovery(
            raw_output, ReasonerOutput, "reasoner", ValidationStrategy.GRACEFUL
        )

    @staticmethod
    def validate_critic_output(raw_output: str, critic_type: str = "critic") -> CriticOutput:
        """Validate and parse critic agent output."""
        return SchemaValidator._validator.validate_with_recovery(
            raw_output, CriticOutput, critic_type, ValidationStrategy.GRACEFUL
        )

    @staticmethod
    def validate_synthesiser_output(raw_output: str) -> SynthesiserOutput:
        """Validate and parse synthesiser agent output."""
        return SchemaValidator._validator.validate_with_recovery(
            raw_output, SynthesiserOutput, "synthesiser", ValidationStrategy.GRACEFUL
        )
    
    @staticmethod
    def get_validation_stats() -> Dict[str, Dict[str, int]]:
        """Get validation statistics."""
        return SchemaValidator._validator.get_validation_stats() 