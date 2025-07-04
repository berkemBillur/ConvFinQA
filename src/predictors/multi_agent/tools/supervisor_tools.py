"""
Supervisor agent tools for ConvFinQA CrewAI implementation.

Provides task decomposition and conversation context management tools.
"""

import json
from typing import Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class TaskDecomposerTool(BaseTool):
    """Tool for decomposing conversational finance questions into structured subtasks."""
    
    name: str = "task_decomposer"
    description: str = "Decompose complex conversational financial queries into structured subtasks for specialist agents"
    
    def _run(self, question: str, conversation_history: str = "") -> str:
        """
        Decompose a financial question into structured subtasks.
        
        Args:
            question: Current financial question to analyse
            conversation_history: Previous conversation context
            
        Returns:
            JSON string with structured task breakdown
        """
        try:
            # Analyse question type and complexity
            task_analysis = {
                "question_type": self._classify_question_type(question),
                "required_data": self._identify_required_data(question),
                "calculation_complexity": self._assess_calculation_complexity(question),
                "reference_resolution_needed": self._check_references(question, conversation_history),
                "subtasks": self._generate_subtasks(question),
                "agent_assignments": self._assign_agents(question)
            }
            
            return json.dumps(task_analysis, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Task decomposition failed: {str(e)}"})
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of financial question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['ratio', 'margin', 'percentage']):
            return "ratio_calculation"
        elif any(word in question_lower for word in ['growth', 'increase', 'decrease', 'change']):
            return "trend_analysis"
        elif any(word in question_lower for word in ['total', 'sum', 'add']):
            return "aggregation"
        elif any(word in question_lower for word in ['compare', 'difference', 'versus']):
            return "comparison"
        elif any(word in question_lower for word in ['lookup', 'what is', 'find']):
            return "data_lookup"
        else:
            return "complex_analysis"
    
    def _identify_required_data(self, question: str) -> List[str]:
        """Identify what financial data is needed."""
        data_types = []
        question_lower = question.lower()
        
        financial_metrics = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash', 'debt', 'sales', 'costs', 'expenses'
        ]
        
        for metric in financial_metrics:
            if metric in question_lower:
                data_types.append(metric)
        
        return data_types
    
    def _assess_calculation_complexity(self, question: str) -> str:
        """Assess the complexity of calculations needed."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['ratio', 'percentage', 'margin']):
            return "medium"
        elif any(word in question_lower for word in ['growth', 'compound', 'average']):
            return "high"
        elif any(word in question_lower for word in ['total', 'sum', 'difference']):
            return "low"
        else:
            return "medium"
    
    def _check_references(self, question: str, conversation_history: str) -> bool:
        """Check if question contains conversational references."""
        references = ['it', 'that', 'this', 'they', 'previous', 'last', 'above', 'below']
        return any(ref in question.lower() for ref in references)
    
    def _generate_subtasks(self, question: str) -> List[str]:
        """Generate structured subtasks for the question."""
        subtasks = []
        question_type = self._classify_question_type(question)
        
        if question_type == "data_lookup":
            subtasks = [
                "extract_relevant_data",
                "validate_data_accuracy"
            ]
        elif question_type == "ratio_calculation":
            subtasks = [
                "extract_numerator_data",
                "extract_denominator_data", 
                "calculate_ratio",
                "validate_calculation"
            ]
        elif question_type == "trend_analysis":
            subtasks = [
                "extract_time_series_data",
                "calculate_changes",
                "analyse_trends",
                "validate_results"
            ]
        else:
            subtasks = [
                "extract_relevant_data",
                "perform_calculations",
                "validate_results"
            ]
        
        return subtasks
    
    def _assign_agents(self, question: str) -> Dict[str, List[str]]:
        """Assign subtasks to appropriate specialist agents."""
        question_type = self._classify_question_type(question)
        
        assignments = {
            "data_extractor": ["extract_relevant_data", "extract_numerator_data", "extract_denominator_data", "extract_time_series_data"],
            "calculations_specialist": ["calculate_ratio", "calculate_changes", "perform_calculations", "analyse_trends"],
            "validator": ["validate_data_accuracy", "validate_calculation", "validate_results"]
        }
        
        return assignments


class ConversationTrackerTool(BaseTool):
    """Tool for tracking and managing conversation context across turns."""
    
    name: str = "conversation_tracker"
    description: str = "Track conversation context and resolve references across multiple turns"
    
    def _run(self, current_question: str, conversation_history: str) -> str:
        """
        Analyse conversation context and track references.
        
        Args:
            current_question: Current question being asked
            conversation_history: Full conversation history
            
        Returns:
            JSON string with conversation analysis
        """
        try:
            context_analysis = {
                "turn_number": self._count_turns(conversation_history),
                "entities_mentioned": self._extract_entities(conversation_history),
                "temporal_references": self._extract_temporal_references(conversation_history),
                "numerical_references": self._extract_numerical_references(conversation_history),
                "context_continuity": self._assess_context_continuity(current_question, conversation_history),
                "reference_resolution": self._resolve_references(current_question, conversation_history)
            }
            
            return json.dumps(context_analysis, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Conversation tracking failed: {str(e)}"})
    
    def _count_turns(self, conversation_history: str) -> int:
        """Count the number of conversation turns."""
        return conversation_history.count("Turn ")
    
    def _extract_entities(self, conversation_history: str) -> List[str]:
        """Extract financial entities mentioned in conversation."""
        entities = []
        
        # Common financial entities
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash', 'debt', 'sales', 'costs', 'expenses',
            'margin', 'ratio', 'growth', 'percentage'
        ]
        
        for term in financial_terms:
            if term in conversation_history.lower():
                entities.append(term)
        
        return list(set(entities))
    
    def _extract_temporal_references(self, conversation_history: str) -> List[str]:
        """Extract temporal references from conversation."""
        temporal_terms = []
        
        # Years, quarters, periods
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        
        years = re.findall(year_pattern, conversation_history)
        quarters = re.findall(quarter_pattern, conversation_history, re.IGNORECASE)
        
        temporal_terms.extend([f"{y[0]}{y[1]}" for y in years])
        temporal_terms.extend(quarters)
        
        return list(set(temporal_terms))
    
    def _extract_numerical_references(self, conversation_history: str) -> List[str]:
        """Extract numerical values mentioned in conversation."""
        import re
        
        # Extract numbers with potential currency or percentage symbols
        number_pattern = r'[\$£€]?[\d,]+\.?\d*[%]?'
        numbers = re.findall(number_pattern, conversation_history)
        
        return list(set(numbers))
    
    def _assess_context_continuity(self, current_question: str, conversation_history: str) -> str:
        """Assess how well current question continues from previous context."""
        current_lower = current_question.lower()
        
        # Check for referential pronouns
        if any(ref in current_lower for ref in ['it', 'that', 'this', 'they']):
            return "high_continuity"
        elif any(ref in current_lower for ref in ['previous', 'last', 'above']):
            return "medium_continuity"
        else:
            return "low_continuity"
    
    def _resolve_references(self, current_question: str, conversation_history: str) -> Dict[str, str]:
        """Attempt to resolve conversational references."""
        resolutions = {}
        current_lower = current_question.lower()
        
        # Simple reference resolution
        if 'it' in current_lower:
            # Find the most recent financial entity mentioned
            entities = self._extract_entities(conversation_history)
            if entities:
                resolutions['it'] = entities[-1]  # Most recent entity
        
        if 'that' in current_lower:
            # Similar logic for 'that'
            numbers = self._extract_numerical_references(conversation_history)
            if numbers:
                resolutions['that'] = numbers[-1]
        
        return resolutions 