"""
Financial KPI Fuzzy Matcher

Implements research-based fuzzy matching for financial KPIs with curated synonym lists
and Levenshtein distance scoring to improve extraction accuracy.

Based on ConvFinQA and FinQA research showing 3-7pp accuracy improvements.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re

try:
    from rapidfuzz import fuzz
except ImportError:
    # Graceful fallback for environments without rapidfuzz
    class fuzz:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            """Fallback implementation"""
            a_lower = a.lower()
            b_lower = b.lower()
            if a_lower == b_lower:
                return 100.0
            if a_lower in b_lower or b_lower in a_lower:
                return 80.0
            return 0.0


@dataclass
class MatchResult:
    """Result of a KPI matching operation."""
    matched_label: str
    confidence: float
    original_query: str
    match_type: str  # "exact", "fuzzy", "synonym"


class FinancialKPIMatcher:
    """
    Advanced financial KPI matcher using fuzzy string matching and domain knowledge.
    
    Implements research-proven techniques for improving ConvFinQA-style extraction:
    - Levenshtein distance with 0.85+ threshold
    - Curated financial synonym lists
    - Unit and scale aware matching
    """
    
    # Curated financial KPI synonyms based on ConvFinQA dataset analysis
    FINANCIAL_SYNONYMS: Dict[str, Set[str]] = {
        # Revenue and Income
        "revenue": {"revenues", "sales", "net sales", "total revenue", "total revenues"},
        "net earnings": {"net income", "net profit", "earnings", "profit", "net earnings"},
        "operating income": {"operating profit", "operating earnings", "ebit", "operating income"},
        "gross profit": {"gross income", "gross earnings", "gross profit"},
        
        # Expenses and Costs
        "repairs and maintenance": {"repair", "maintenance", "r&m", "repairs", "maintenance expense", 
                                   "repair and maintenance", "repairs & maintenance", "maintenance costs"},
        "operating expenses": {"operating costs", "opex", "operational expenses", "operating expense"},
        "cost of goods sold": {"cogs", "cost of sales", "cost of revenue", "cost of goods sold"},
        "depreciation": {"depreciation expense", "depreciation and amortization", "d&a"},
        
        # Financial Position
        "total assets": {"assets", "total asset", "asset total"},
        "total liabilities": {"liabilities", "total liability", "liability total"},
        "stockholders equity": {"shareholders equity", "equity", "total equity", "stockholder equity"},
        "cash and cash equivalents": {"cash", "cash equivalents", "cash and equivalents"},
        
        # Per Share Metrics
        "earnings per share": {"eps", "basic eps", "diluted eps", "earnings per share"},
        "book value per share": {"book value", "bvps", "book value per share"},
        "weighted average exercise price": {"exercise price", "average exercise price", "weighted exercise price"},
        
        # Ratios and Percentages
        "return on assets": {"roa", "return on asset", "asset return"},
        "return on equity": {"roe", "return on equity", "equity return"},
        "profit margin": {"net margin", "profit margin", "net profit margin"},
        
        # Cash Flow
        "cash flow": {"cash flows", "net cash flow", "operating cash flow"},
        "free cash flow": {"fcf", "free cash flows", "unlevered free cash flow"},
        
        # Growth and Changes
        "increase": {"growth", "rise", "gain", "improvement", "uptick"},
        "decrease": {"decline", "drop", "fall", "reduction", "downturn"},
        "change": {"difference", "variation", "delta", "shift"},
    }
    
    # Scale indicators for unit-aware matching
    SCALE_INDICATORS: Dict[str, float] = {
        "thousand": 1_000,
        "thousands": 1_000,
        "million": 1_000_000,
        "millions": 1_000_000,
        "billion": 1_000_000_000,
        "billions": 1_000_000_000,
        "trillion": 1_000_000_000_000,
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
    }
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the financial KPI matcher.
        
        Args:
            similarity_threshold: Minimum similarity score for fuzzy matches (0.85+ recommended by research)
        """
        self.similarity_threshold = similarity_threshold
        
        # Build reverse synonym lookup
        self._synonym_lookup: Dict[str, str] = {}
        for canonical, synonyms in self.FINANCIAL_SYNONYMS.items():
            self._synonym_lookup[canonical] = canonical
            for synonym in synonyms:
                self._synonym_lookup[synonym] = canonical
    
    def find_best_match(
        self, 
        query_kpi: str, 
        available_labels: List[str],
        context: Optional[str] = None
    ) -> Optional[MatchResult]:
        """
        Find the best matching KPI label using multiple strategies.
        
        Args:
            query_kpi: The financial KPI term to match
            available_labels: List of available row labels in the financial table
            context: Additional context (question text) for better matching
            
        Returns:
            MatchResult with the best match, or None if no good match found
        """
        if not available_labels:
            return None
            
        query_clean = self._clean_text(query_kpi)
        
        # Strategy 1: Exact match
        exact_match = self._find_exact_match(query_clean, available_labels)
        if exact_match:
            return MatchResult(
                matched_label=exact_match,
                confidence=1.0,
                original_query=query_kpi,
                match_type="exact"
            )
        
        # Strategy 2: Synonym-based match
        synonym_match = self._find_synonym_match(query_clean, available_labels)
        if synonym_match:
            return synonym_match
            
        # Strategy 3: Fuzzy match with Levenshtein distance
        fuzzy_match = self._find_fuzzy_match(query_clean, available_labels)
        if fuzzy_match:
            return fuzzy_match
            
        # Strategy 4: Contextual partial match (fallback)
        if context:
            partial_match = self._find_contextual_match(query_clean, available_labels, context)
            if partial_match:
                return partial_match
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for matching."""
        # Convert to lowercase and remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove common financial formatting
        cleaned = re.sub(r'[,$%()]', '', cleaned)
        
        # Normalize common abbreviations
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('+', 'and')
        
        return cleaned
    
    def _find_exact_match(self, query: str, labels: List[str]) -> Optional[str]:
        """Find exact text match."""
        query_lower = query.lower()
        for label in labels:
            if self._clean_text(label).lower() == query_lower:
                return label
        return None
    
    def _find_synonym_match(self, query: str, labels: List[str]) -> Optional[MatchResult]:
        """Find match using financial synonym dictionary."""
        # Check if query maps to a canonical term
        canonical_query = self._synonym_lookup.get(query)
        if not canonical_query:
            return None
            
        # Look for labels that match the canonical term or its synonyms
        synonyms = self.FINANCIAL_SYNONYMS.get(canonical_query, set())
        all_variants = {canonical_query} | synonyms
        
        for label in labels:
            label_clean = self._clean_text(label)
            for variant in all_variants:
                if variant in label_clean or label_clean in variant:
                    return MatchResult(
                        matched_label=label,
                        confidence=0.95,
                        original_query=query,
                        match_type="synonym"
                    )
        return None
    
    def _find_fuzzy_match(self, query: str, labels: List[str]) -> Optional[MatchResult]:
        """Find match using fuzzy string similarity."""
        best_score = 0.0
        best_label = None
        
        for label in labels:
            label_clean = self._clean_text(label)
            
            # Use token set ratio for better handling of word order differences
            score = fuzz.token_set_ratio(query, label_clean) / 100.0
            
            # Boost score if partial words match
            if any(word in label_clean for word in query.split() if len(word) > 2):
                score += 0.1
                
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_label = label
        
        if best_label:
            return MatchResult(
                matched_label=best_label,
                confidence=best_score,
                original_query=query,
                match_type="fuzzy"
            )
        return None
    
    def _find_contextual_match(
        self, 
        query: str, 
        labels: List[str], 
        context: str
    ) -> Optional[MatchResult]:
        """Find match using additional context clues."""
        context_clean = self._clean_text(context)
        
        # Extract financial context keywords
        context_keywords = set(context_clean.split())
        
        best_score = 0.0
        best_label = None
        
        for label in labels:
            label_clean = self._clean_text(label)
            label_words = set(label_clean.split())
            
            # Score based on keyword overlap
            overlap = len(context_keywords & label_words)
            if overlap > 0:
                # Normalize by label length to prefer more specific matches
                score = overlap / len(label_words) if label_words else 0
                
                if score > best_score and score >= 0.3:  # Lower threshold for contextual
                    best_score = score
                    best_label = label
        
        if best_label:
            return MatchResult(
                matched_label=best_label,
                confidence=best_score * 0.7,  # Lower confidence for contextual matches
                original_query=query,
                match_type="contextual"
            )
        return None
    
    def extract_scale_info(self, text: str) -> Tuple[Optional[float], str]:
        """
        Extract scale/unit information from text.
        
        Returns:
            Tuple of (scale_multiplier, cleaned_text)
        """
        text_lower = text.lower()
        
        for indicator, multiplier in self.SCALE_INDICATORS.items():
            if indicator in text_lower:
                # Remove the scale indicator from text
                cleaned = re.sub(rf'\b{re.escape(indicator)}\b', '', text_lower)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                return multiplier, cleaned
        
        return None, text


# Global instance for easy import
financial_matcher = FinancialKPIMatcher() 