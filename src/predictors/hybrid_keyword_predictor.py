"""Hybrid Keyword-Heuristic benchmark predictor for ConvFinQA."""

from typing import Dict, List, Union, Optional, Tuple
import re
from dataclasses import dataclass

try:
    from ..data.models import ConvFinQARecord
    from ..data.utils import extract_value_candidates, ValueCandidate
    from ..utils.text_processing import QuestionClassifier, OperationMatch
    from ..utils.config import get_config
    from ..evaluation.executor import execute_dsl_program
except ImportError:
    from data.models import ConvFinQARecord
    from data.utils import extract_value_candidates, ValueCandidate
    from utils.text_processing import QuestionClassifier, OperationMatch
    from utils.config import get_config
    from evaluation.executor import execute_dsl_program


@dataclass
class PredictionResult:
    """Result of a prediction with confidence and reasoning."""
    answer: Union[float, str]
    dsl_program: str
    confidence: float
    operation_type: str
    reasoning: str


class HybridKeywordPredictor:
    """Hybrid keyword-heuristic predictor implementing our benchmark approach.
    
    This predictor follows a 4-component pipeline:
    1. Question Classification - keyword-based operation detection
    2. Value Extraction - heuristic scoring of table candidates
    3. DSL Generation - template-based program synthesis  
    4. Fallback Strategy - progressive degradation for low confidence
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialise the hybrid predictor.
        
        Args:
            config_path: Optional path to configuration file.
        """
        self.config = get_config(config_path)
        
        # Initialise question classifier with operation keywords
        hybrid_config = self.config.get('models.hybrid_keyword', {})
        operation_keywords = hybrid_config.get('question_classification', {}).get('operation_keywords', {})
        confidence_threshold = hybrid_config.get('question_classification', {}).get('confidence_threshold', 0.5)
        
        self.question_classifier = QuestionClassifier(
            operation_keywords=operation_keywords,
            confidence_threshold=confidence_threshold
        )
        
        # DSL templates for different operations
        self.dsl_templates = hybrid_config.get('dsl_generation', {}).get('templates', {})
        
        # Confidence thresholds for fallback strategy
        self.high_confidence_threshold = 0.7
        self.medium_confidence_threshold = 0.3
        
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Union[float, str]:
        """Predict answer for a single conversation turn.
        
        Args:
            record: ConvFinQA record containing document and conversation.
            turn_index: Index of current turn (0-based).
            conversation_history: Previous turns with questions and answers.
            
        Returns:
            Predicted answer for the turn.
        """
        if turn_index >= len(record.dialogue.conv_questions):
            return "Turn index out of range"
        
        question = record.dialogue.conv_questions[turn_index]
        
        try:
            # Get full prediction with reasoning
            prediction_result = self._predict_with_reasoning(
                record, question, turn_index, conversation_history
            )
            
            # Execute the DSL program to get final answer
            if prediction_result.dsl_program:
                answer = execute_dsl_program(prediction_result.dsl_program)
                return answer if isinstance(answer, (int, float)) else str(answer)
            else:
                return prediction_result.answer
                
        except Exception as e:
            # Fallback to simple table value on any error
            return self._fallback_to_first_table_value(record)
    
    def _predict_with_reasoning(
        self,
        record: ConvFinQARecord,
        question: str,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> PredictionResult:
        """Perform full prediction with detailed reasoning.
        
        This is the main implementation of our 4-component pipeline.
        """
        # Component 1: Question Classification
        operation_match = self._classify_question(question, conversation_history)
        
        # Component 2: Value Extraction
        value_candidates = self._extract_values(record, question, conversation_history)
        
        # Component 3: DSL Generation
        dsl_program, confidence = self._generate_dsl(
            operation_match, value_candidates, turn_index, conversation_history
        )
        
        # Component 4: Fallback Strategy
        if confidence >= self.high_confidence_threshold:
            # High confidence: use predicted DSL
            return PredictionResult(
                answer=0.0,  # Will be computed by DSL execution
                dsl_program=dsl_program,
                confidence=confidence,
                operation_type=operation_match.operation_type if operation_match else "unknown",
                reasoning="High confidence prediction using detected operation and extracted values"
            )
        elif confidence >= self.medium_confidence_threshold:
            # Medium confidence: simple lookup of best value
            best_value = value_candidates[0].value if value_candidates else 0.0
            return PredictionResult(
                answer=best_value,
                dsl_program=str(best_value),
                confidence=confidence,
                operation_type="lookup",
                reasoning="Medium confidence fallback to best table value"
            )
        else:
            # Low confidence: return first relevant table value
            fallback_value = self._fallback_to_first_table_value(record)
            return PredictionResult(
                answer=fallback_value,
                dsl_program=str(fallback_value),
                confidence=confidence,
                operation_type="fallback",
                reasoning="Low confidence fallback to first table value"
            )
    
    def _classify_question(
        self, 
        question: str, 
        conversation_history: List[Dict[str, str]]
    ) -> Optional[OperationMatch]:
        """Component 1: Classify question into operation type."""
        # Check for reference keywords first
        hybrid_config = self.config.get('models.hybrid_keyword', {})
        reference_keywords = hybrid_config.get('question_classification', {}).get('reference_keywords', [])
        question_lower = question.lower()
        
        # If reference detected, try to inherit operation from previous turn
        if any(keyword in question_lower for keyword in reference_keywords):
            if conversation_history:
                # For now, assume same operation as previous turn
                # This is a simplified heuristic that could be improved
                return OperationMatch(
                    operation_type="lookup",  # Default for references
                    confidence=0.6,
                    matched_keywords=["reference"],
                    context=question_lower
                )
        
        # Use question classifier for direct operation detection
        return self.question_classifier.classify_question(question)
    
    def _extract_values(
        self,
        record: ConvFinQARecord,
        question: str,
        conversation_history: List[Dict[str, str]]
    ) -> List[ValueCandidate]:
        """Component 2: Extract and score value candidates from table."""
        hybrid_config = self.config.get('models.hybrid_keyword', {})
        financial_keywords = hybrid_config.get('value_extraction', {}).get('financial_keywords', [])
        max_candidates = hybrid_config.get('value_extraction', {}).get('max_candidates', 5)
        
        # Use existing utility function
        candidates = extract_value_candidates(
            document=record.doc,
            question=question,
            financial_keywords=financial_keywords,
            max_candidates=max_candidates
        )
        
        return candidates
    
    def _generate_dsl(
        self,
        operation_match: Optional[OperationMatch],
        value_candidates: List[ValueCandidate],
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, float]:
        """Component 3: Generate DSL program using templates."""
        if not operation_match or not value_candidates:
            # No operation detected or no values found
            if value_candidates:
                return str(value_candidates[0].value), 0.2
            return "0.0", 0.1
        
        operation_type = operation_match.operation_type
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(operation_match, value_candidates)
        
        # Generate DSL based on operation type
        if operation_type == "lookup":
            # Simple value lookup
            return str(value_candidates[0].value), confidence
            
        elif operation_type in ["addition", "subtraction", "multiplication", "division"]:
            # Binary operations need two values
            if len(value_candidates) >= 2:
                template = self.dsl_templates[operation_type]
                dsl = template.format(
                    value1=value_candidates[0].value,
                    value2=value_candidates[1].value
                )
                return dsl, confidence
            else:
                # Fallback to simple lookup if insufficient values
                return str(value_candidates[0].value), confidence * 0.5
        
        else:
            # Unknown operation, fallback to lookup
            return str(value_candidates[0].value), confidence * 0.3
    
    def _calculate_confidence(
        self,
        operation_match: Optional[OperationMatch],
        value_candidates: List[ValueCandidate]
    ) -> float:
        """Calculate overall prediction confidence."""
        if not operation_match or not value_candidates:
            return 0.1
        
        # Use weights from config
        hybrid_config = self.config.get('models.hybrid_keyword', {})
        weights_dict = hybrid_config.get('value_extraction', {}).get('scoring_weights', {})
        
        # Create a simple object to hold weights
        class Weights:
            def __init__(self, weights_dict):
                self.header_match = weights_dict.get('header_match', 0.4)
                self.keyword_proximity = weights_dict.get('keyword_proximity', 0.3) 
                self.question_alignment = weights_dict.get('question_alignment', 0.2)
                self.financial_relevance = weights_dict.get('financial_relevance', 0.1)
        
        weights = Weights(weights_dict)
        
        # Operation confidence
        keyword_score = operation_match.confidence
        
        # Value extraction quality
        value_score = value_candidates[0].score if value_candidates else 0.0
        
        # Simple context clarity (could be enhanced)
        context_score = 0.5 if len(value_candidates) >= 2 else 0.3
        
        # Combined confidence using config weights
        confidence = (
            keyword_score * weights.header_match +
            value_score * weights.keyword_proximity +
            context_score * weights.question_alignment
        )
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _fallback_to_first_table_value(self, record: ConvFinQARecord) -> float:
        """Fallback strategy: return first numeric value from table."""
        if not record.doc.table:
            return 0.0
        
        for col_data in record.doc.table.values():
            for value in col_data.values():
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Try to extract numeric value
                    try:
                        cleaned = value.replace(',', '').replace('$', '').strip()
                        if cleaned.startswith('(') and cleaned.endswith(')'):
                            cleaned = '-' + cleaned[1:-1]
                        return float(cleaned)
                    except ValueError:
                        continue
        
        return 0.0


# Trade-offs and limitations:
# 1. Reference resolution is simplified - only looks at immediate context
# 2. Value pairing for binary operations uses simple ranking rather than semantic matching
# 3. Confidence calculation could be more sophisticated with domain-specific heuristics
# 4. Error handling prioritises robustness over precision in edge cases 