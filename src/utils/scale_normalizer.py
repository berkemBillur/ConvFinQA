"""
Scale-Aware Financial Checker

Implements automatic scale detection and conversion for financial calculations
to fix common magnitude errors in ConvFinQA-style tasks.

Based on research showing 1-2pp accuracy improvements on ratio/percentage questions.
"""

from __future__ import annotations
import re
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import math


@dataclass
class ScaleInfo:
    """Information about detected scale in text or values."""
    original_text: str
    detected_scale: Optional[str]
    scale_multiplier: Optional[float]
    is_percentage: bool
    cleaned_text: str


class ScaleNormalizer:
    """
    Advanced scale detection and normalization for financial calculations.
    
    Handles common magnitude errors like:
    - "$ 3.3 bn ÷ $ 988 m" should output 0.296 then × 100 → 29.6%
    - Million vs billion unit mismatches
    - Percentage vs decimal ratio confusion
    """
    
    # Scale indicators with their multipliers
    SCALE_PATTERNS = {
        # Explicit scale words
        r'\b(?:thousand|thousands)\b': 1_000,
        r'\b(?:million|millions)\b': 1_000_000,
        r'\b(?:billion|billions)\b': 1_000_000_000,
        r'\b(?:trillion|trillions)\b': 1_000_000_000_000,
        
        # Abbreviated forms
        r'\b(?:k|K)\b': 1_000,
        r'\b(?:m|M)\b(?!\s*(?:onth|arch))': 1_000_000,  # Avoid "month", "March"
        r'\b(?:b|B)\b(?!\s*(?:illion|y))': 1_000_000_000,  # Avoid "billion", "by"
        r'\b(?:t|T)\b(?!\s*(?:rillion|o))': 1_000_000_000_000,  # Avoid "trillion", "to"
        
        # With currency symbols
        r'\$\s*[\d,.]+\s*(?:thousand|k)': 1_000,
        r'\$\s*[\d,.]+\s*(?:million|m)': 1_000_000,
        r'\$\s*[\d,.]+\s*(?:billion|b)': 1_000_000_000,
    }
    
    # Percentage indicators
    PERCENTAGE_PATTERNS = [
        r'%',
        r'\bpercent\b',
        r'\bpercentage\b',
        r'\bpct\b',
        r'\bbps\b',  # basis points
    ]
    
    # Common financial ratio patterns that should be percentages
    RATIO_TO_PERCENTAGE_PATTERNS = [
        r'(?:percentage|%) (?:of|in)',
        r'(?:ratio|rate) (?:of|to)',
        r'(?:return on|margin)',
        r'(?:change|growth|decline) (?:in|of)',
    ]
    
    def __init__(self):
        """Initialize the scale normalizer."""
        self.compiled_scale_patterns = {
            re.compile(pattern, re.IGNORECASE): multiplier 
            for pattern, multiplier in self.SCALE_PATTERNS.items()
        }
        
        self.compiled_percentage_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.PERCENTAGE_PATTERNS
        ]
        
        self.compiled_ratio_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.RATIO_TO_PERCENTAGE_PATTERNS
        ]
    
    def analyze_scale(self, text: str) -> ScaleInfo:
        """
        Analyze text for scale indicators and percentage context.
        
        Args:
            text: Text to analyze for scale information
            
        Returns:
            ScaleInfo object with detected scale information
        """
        text_lower = text.lower()
        
        # Detect scale
        detected_scale = None
        scale_multiplier = None
        
        for pattern, multiplier in self.compiled_scale_patterns.items():
            if pattern.search(text):
                detected_scale = pattern.pattern
                scale_multiplier = multiplier
                break
        
        # Detect percentage context
        is_percentage = any(
            pattern.search(text) for pattern in self.compiled_percentage_patterns
        )
        
        # Also check for ratio-to-percentage patterns
        if not is_percentage:
            is_percentage = any(
                pattern.search(text) for pattern in self.compiled_ratio_patterns
            )
        
        # Clean text by removing scale indicators
        cleaned_text = text
        if detected_scale:
            cleaned_text = re.sub(detected_scale, '', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return ScaleInfo(
            original_text=text,
            detected_scale=detected_scale,
            scale_multiplier=scale_multiplier,
            is_percentage=is_percentage,
            cleaned_text=cleaned_text
        )
    
    def normalize_calculation_result(
        self, 
        result: Union[float, str], 
        question_context: str,
        calculation_type: Optional[str] = None
    ) -> float:
        """
        Apply scale normalization to calculation results.
        
        Args:
            result: Raw calculation result
            question_context: Original question for context analysis
            calculation_type: Type of calculation performed (optional)
            
        Returns:
            Normalized result with appropriate scale
        """
        try:
            numeric_result = float(result)
        except (ValueError, TypeError):
            return 0.0
        
        # Analyze question context
        context_info = self.analyze_scale(question_context)
        
        # Apply normalization rules
        normalized = self._apply_normalization_rules(
            numeric_result, 
            context_info, 
            calculation_type
        )
        
        return normalized
    
    def _apply_normalization_rules(
        self, 
        result: float, 
        context_info: ScaleInfo,
        calculation_type: Optional[str]
    ) -> float:
        """Apply specific normalization rules based on context analysis."""
        
        # CONSERVATIVE APPROACH: Only apply corrections when confidence is very high
        
        # Rule 1: Percentage conversion for ratios (ONLY for clear percentage questions)
        if context_info.is_percentage and result < 1.0 and result > 0:
            # Only convert if question explicitly asks for percentage
            if any(word in context_info.original_text.lower() for word in ["percentage", "percent", "%"]):
                return result * 100
        
        # Rule 2: Handle division results that are clearly ratios
        if calculation_type and "divide" in calculation_type.lower():
            if result > 100 and context_info.is_percentage:
                # Only if explicitly asking for percentage AND result is unreasonably large
                if "percentage" in context_info.original_text.lower() and result > 1000:
                    return result / 100
        
        # Rule 3: DISABLED - Scale mismatch detection was too aggressive
        # The original data might already be in correct scale
        
        # Rule 4: Only apply percentage conversion for explicit percentage questions
        if "percentage" in context_info.original_text.lower():
            # If asking for percentage but got small decimal
            if 0 < result < 1:
                return result * 100
        
        # CONSERVATIVE: Return original result unless we're very confident
        return result
    
    def detect_unit_mismatch(
        self, 
        dividend_text: str, 
        divisor_text: str,
        result: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect unit mismatches in division operations.
        
        Args:
            dividend_text: Text describing the dividend
            divisor_text: Text describing the divisor  
            result: The division result
            
        Returns:
            Tuple of (has_mismatch, correction_suggestion)
        """
        dividend_info = self.analyze_scale(dividend_text)
        divisor_info = self.analyze_scale(divisor_text)
        
        # Check for scale mismatches
        if (dividend_info.scale_multiplier and divisor_info.scale_multiplier and
            dividend_info.scale_multiplier != divisor_info.scale_multiplier):
            
            expected_ratio = dividend_info.scale_multiplier / divisor_info.scale_multiplier
            
            # If result is way off expected ratio, suggest correction
            if abs(result - expected_ratio) > abs(expected_ratio * 0.5):
                suggestion = f"Scale mismatch detected: {dividend_info.detected_scale} ÷ {divisor_info.detected_scale}"
                return True, suggestion
        
        return False, None
    
    def format_result_with_context(
        self, 
        result: float, 
        context_info: ScaleInfo,
        precision: int = 5
    ) -> str:
        """
        Format result appropriately based on context.
        
        Args:
            result: Numerical result to format
            context_info: Context information about expected format
            precision: Decimal places for formatting
            
        Returns:
            Formatted result string
        """
        # Handle very small percentages
        if context_info.is_percentage and abs(result) < 1:
            return f"{result:.{precision}f}"
        
        # Handle regular percentages  
        elif context_info.is_percentage:
            return f"{result:.2f}"
        
        # Handle large financial numbers
        elif abs(result) > 1_000_000:
            return f"{result:.1f}"
        
        # Handle regular financial values
        else:
            return f"{result:.{min(precision, 3)}f}"


# Global instance for easy import
scale_normalizer = ScaleNormalizer() 