"""Text processing utilities for ConvFinQA question analysis."""

from typing import Dict, List, Tuple, Optional, Set, Any
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class OperationMatch:
    """Represents a matched operation with confidence score."""
    operation_type: str
    confidence: float
    matched_keywords: List[str]
    context: str


class QuestionClassifier:
    """Classifies questions into operation types based on keyword patterns."""
    
    def __init__(self, operation_keywords: Dict[str, List[str]], confidence_threshold: float = 0.5):
        """Initialise the question classifier.
        
        Args:
            operation_keywords: Dictionary mapping operation types to keyword lists.
            confidence_threshold: Minimum confidence required for classification.
        """
        self.operation_keywords = operation_keywords
        self.confidence_threshold = confidence_threshold
        
        # Build compiled patterns for efficiency
        self._patterns = self._compile_patterns()
    
    def classify_question(self, question: str) -> Optional[OperationMatch]:
        """Classify a question into an operation type.
        
        Args:
            question: Question text to classify.
            
        Returns:
            OperationMatch if confident classification found, None otherwise.
        """
        question_lower = question.lower()
        operation_scores = defaultdict(list)
        
        # Score each operation type
        for operation_type, keywords in self.operation_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    # Calculate keyword strength (longer keywords = higher score)
                    strength = len(keyword.split()) * 0.3 + 0.1
                    operation_scores[operation_type].append({
                        'keyword': keyword,
                        'strength': strength
                    })
        
        # Find best operation match
        best_operation = None
        best_confidence = 0.0
        best_keywords = []
        
        for operation_type, matches in operation_scores.items():
            if matches:
                # Calculate confidence as sum of keyword strengths
                confidence = sum(match['strength'] for match in matches)
                keywords = [match['keyword'] for match in matches]
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_operation = operation_type
                    best_keywords = keywords
        
        # Return match if above threshold
        if best_confidence >= self.confidence_threshold and best_operation:
            return OperationMatch(
                operation_type=best_operation,
                confidence=best_confidence,
                matched_keywords=best_keywords,
                context=question_lower
            )
        
        return None
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient matching."""
        patterns = {}
        for operation_type, keywords in self.operation_keywords.items():
            # Create pattern that matches any of the keywords
            escaped_keywords = [re.escape(keyword) for keyword in keywords]
            pattern = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
            patterns[operation_type] = re.compile(pattern, re.IGNORECASE)
        return patterns


class ReferenceDetector:
    """Detects references to previous conversation turns."""
    
    def __init__(self, reference_keywords: List[str]):
        """Initialise the reference detector.
        
        Args:
            reference_keywords: Keywords that indicate references to previous turns.
        """
        self.reference_keywords = reference_keywords
        self._pattern = self._compile_reference_pattern()
    
    def detect_references(self, question: str, conversation_history: List[str]) -> Dict[str, Any]:
        """Detect if question references previous conversation turns.
        
        Args:
            question: Current question text.
            conversation_history: List of previous questions/answers.
            
        Returns:
            Dictionary with reference information.
        """
        question_lower = question.lower()
        
        result = {
            'has_reference': False,
            'reference_type': None,
            'likely_target': None,
            'confidence': 0.0
        }
        
        # Check for reference keywords
        found_keywords = []
        for keyword in self.reference_keywords:
            if keyword in question_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            result['has_reference'] = True
            result['reference_type'] = 'explicit'
            result['confidence'] = min(len(found_keywords) * 0.3, 1.0)
            
            # Simple heuristic: most recent reference is most likely
            if conversation_history:
                result['likely_target'] = len(conversation_history) - 1
        
        # Check for implicit references (numbers from previous answers)
        if not result['has_reference'] and conversation_history:
            previous_numbers = self._extract_numbers_from_history(conversation_history)
            question_numbers = self._extract_numbers(question)
            
            # If question contains numbers from previous answers, likely reference
            if any(num in previous_numbers for num in question_numbers):
                result['has_reference'] = True
                result['reference_type'] = 'implicit'
                result['confidence'] = 0.4
                result['likely_target'] = len(conversation_history) - 1
        
        return result
    
    def _compile_reference_pattern(self) -> re.Pattern:
        """Compile regex pattern for reference detection."""
        escaped_keywords = [re.escape(keyword) for keyword in self.reference_keywords]
        pattern = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def _extract_numbers(self, text: str) -> Set[float]:
        """Extract numbers from text."""
        numbers = set()
        pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                number = float(match.replace(',', ''))
                numbers.add(number)
            except ValueError:
                continue
        
        return numbers
    
    def _extract_numbers_from_history(self, history: List[str]) -> Set[float]:
        """Extract all numbers from conversation history."""
        all_numbers = set()
        for text in history:
            all_numbers.update(self._extract_numbers(text))
        return all_numbers


def extract_keywords_from_question(question: str, target_keywords: List[str]) -> List[Tuple[str, int]]:
    """Extract target keywords from question with their positions.
    
    Args:
        question: Question text to analyse.
        target_keywords: List of keywords to search for.
        
    Returns:
        List of (keyword, position) tuples for found keywords.
    """
    question_lower = question.lower()
    found_keywords = []
    
    for keyword in target_keywords:
        keyword_lower = keyword.lower()
        start = 0
        
        while True:
            pos = question_lower.find(keyword_lower, start)
            if pos == -1:
                break
            
            # Check if it's a whole word match
            if (pos == 0 or not question_lower[pos-1].isalnum()) and \
               (pos + len(keyword_lower) == len(question_lower) or not question_lower[pos + len(keyword_lower)].isalnum()):
                found_keywords.append((keyword, pos))
            
            start = pos + 1
    
    return found_keywords


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple word overlap similarity between two texts.
    
    Args:
        text1: First text.
        text2: Second text.
        
    Returns:
        Similarity score between 0 and 1.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def normalise_number_text(text: str) -> str:
    """Normalise number representations in text.
    
    Args:
        text: Text containing numbers.
        
    Returns:
        Text with normalised number formats.
    """
    # Remove currency symbols
    text = re.sub(r'[$£€¥]', '', text)
    
    # Handle parentheses for negative numbers
    text = re.sub(r'\((\d+(?:,\d{3})*(?:\.\d+)?)\)', r'-\1', text)
    
    # Normalise thousands separators
    text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)
    
    return text


"""
Trade-offs and Limitations:

1. **Question Classification**: Uses simple keyword matching rather than ML models.
   - Pro: Fast, predictable, easy to debug and extend
   - Con: May miss nuanced questions or new phrasings

2. **Reference Detection**: Basic heuristics for conversation context.
   - Pro: Handles common cases effectively
   - Con: May miss complex multi-turn dependencies

3. **Text Similarity**: Simple word overlap rather than semantic similarity.
   - Pro: No additional dependencies, fast computation
   - Con: May miss semantically similar but differently worded content

4. **Keyword Matching**: Case-insensitive exact matches.
   - Pro: Reliable for known patterns
   - Con: Sensitive to spelling variations and synonyms

Design decisions prioritise simplicity and reliability over sophistication,
following the principle that a working baseline enables iterative improvement.
""" 