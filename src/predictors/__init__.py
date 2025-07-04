"""Predictor implementations for ConvFinQA."""

# Import existing predictors from evaluation module for compatibility
try:
    from ..evaluation.evaluator import Predictor, GroundTruthPredictor, SimpleBaselinePredictor
except ImportError:
    from evaluation.evaluator import Predictor, GroundTruthPredictor, SimpleBaselinePredictor

from .hybrid_keyword_predictor import HybridKeywordPredictor

# CrewAI predictor - optional import with graceful fallback
try:
    from .multi_agent_predictor import ConvFinQAMultiAgentPredictor
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

if CREWAI_AVAILABLE:
    __all__ = [
        'Predictor',
        'GroundTruthPredictor', 
        'SimpleBaselinePredictor',
        'HybridKeywordPredictor',
        'ConvFinQAMultiAgentPredictor',
    ]
else:
    __all__ = [
        'Predictor',
        'GroundTruthPredictor', 
        'SimpleBaselinePredictor',
        'HybridKeywordPredictor',
    ] 