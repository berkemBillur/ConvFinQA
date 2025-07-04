"""
Data extraction tools for ConvFinQA CrewAI implementation.

Provides intelligent table filtering, reference resolution, and temporal parsing tools.
"""

import json
import re
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool


class TableExtractionTool(BaseTool):
    """Tool for intelligent financial table data extraction."""
    
    name: str = "table_extractor"
    description: str = "Extract relevant financial data from tables based on question context"
    
    def _run(self, documents: str, question: str, entities: str = "") -> str:
        """
        Extract relevant financial data from documents.
        
        Args:
            documents: Financial documents containing tables
            question: Current question requiring data
            entities: Relevant entities to focus extraction
            
        Returns:
            JSON string with extracted data
        """
        try:
            extraction_result = {
                "relevant_tables": self._identify_relevant_tables(documents, question),
                "extracted_values": self._extract_numerical_values(documents, question),
                "column_mappings": self._map_relevant_columns(documents, question),
                "row_selections": self._select_relevant_rows(documents, question),
                "data_quality": self._assess_data_quality(documents)
            }
            
            return json.dumps(extraction_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Table extraction failed: {str(e)}"})
    
    def _identify_relevant_tables(self, documents: str, question: str) -> List[str]:
        """Identify which tables are relevant to the question."""
        relevant_tables = []
        question_lower = question.lower()
        
        # Split documents by potential table headers
        doc_sections = documents.split('\n\n')
        
        for i, section in enumerate(doc_sections):
            section_lower = section.lower()
            
            # Check if section contains financial keywords from question
            question_keywords = self._extract_financial_keywords(question)
            
            if any(keyword in section_lower for keyword in question_keywords):
                relevant_tables.append(f"Table_{i}")
        
        return relevant_tables
    
    def _extract_numerical_values(self, documents: str, question: str) -> Dict[str, str]:
        """Extract numerical values relevant to the question."""
        values = {}
        
        # Extract financial keywords from question
        keywords = self._extract_financial_keywords(question)
        
        # Pattern to match financial values
        value_pattern = r'[\$£€]?[\d,]+\.?\d*[%BMK]?'
        
        for keyword in keywords:
            # Look for lines containing both keyword and numerical values
            lines = documents.split('\n')
            for line in lines:
                if keyword.lower() in line.lower():
                    matches = re.findall(value_pattern, line)
                    if matches:
                        values[keyword] = matches
        
        return values
    
    def _extract_financial_keywords(self, question: str) -> List[str]:
        """Extract financial keywords from question."""
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash', 'debt', 'sales', 'costs', 'expenses',
            'margin', 'ratio', 'growth', 'total', 'net'
        ]
        
        question_lower = question.lower()
        found_terms = []
        
        for term in financial_terms:
            if term in question_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _map_relevant_columns(self, documents: str, question: str) -> Dict[str, List[str]]:
        """Map relevant column headers for table extraction."""
        column_mappings = {}
        
        # Extract temporal references from question
        temporal_refs = self._extract_temporal_references(question)
        
        # Look for table headers
        lines = documents.split('\n')
        for line in lines:
            # Simple heuristic: lines with multiple separated values might be headers
            if '\t' in line or '|' in line:
                potential_headers = re.split(r'[\t|]', line)
                potential_headers = [h.strip() for h in potential_headers if h.strip()]
                
                # Check if headers match temporal references or financial terms
                relevant_headers = []
                for header in potential_headers:
                    if any(ref in header for ref in temporal_refs) or \
                       any(term in header.lower() for term in self._extract_financial_keywords(question)):
                        relevant_headers.append(header)
                
                if relevant_headers:
                    column_mappings[line[:50]] = relevant_headers  # Use first 50 chars as key
        
        return column_mappings
    
    def _extract_temporal_references(self, question: str) -> List[str]:
        """Extract temporal references from question."""
        temporal_refs = []
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, question)
        temporal_refs.extend([f"{y[0]}{y[1]}" for y in years])
        
        # Quarters
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        quarters = re.findall(quarter_pattern, question, re.IGNORECASE)
        temporal_refs.extend(quarters)
        
        # Relative terms
        relative_terms = ['previous', 'last', 'current', 'next', 'prior']
        for term in relative_terms:
            if term in question.lower():
                temporal_refs.append(term)
        
        return temporal_refs
    
    def _select_relevant_rows(self, documents: str, question: str) -> List[str]:
        """Select relevant table rows based on question context."""
        relevant_rows = []
        
        # Extract entities and temporal references
        keywords = self._extract_financial_keywords(question)
        temporal_refs = self._extract_temporal_references(question)
        
        lines = documents.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains relevant keywords or temporal references
            if any(keyword in line_lower for keyword in keywords) or \
               any(ref in line for ref in temporal_refs):
                relevant_rows.append(line.strip())
        
        return relevant_rows
    
    def _assess_data_quality(self, documents: str) -> Dict[str, Any]:
        """Assess the quality and completeness of data."""
        quality_assessment = {
            "total_lines": len(documents.split('\n')),
            "numerical_values_found": len(re.findall(r'[\d,]+\.?\d*', documents)),
            "missing_values_indicators": documents.count('N/A') + documents.count('null') + documents.count('-'),
            "data_completeness": "high"  # Simplified assessment
        }
        
        # Simple completeness heuristic
        if quality_assessment["missing_values_indicators"] > quality_assessment["numerical_values_found"] * 0.1:
            quality_assessment["data_completeness"] = "medium"
        if quality_assessment["missing_values_indicators"] > quality_assessment["numerical_values_found"] * 0.3:
            quality_assessment["data_completeness"] = "low"
        
        return quality_assessment


class ReferenceResolverTool(BaseTool):
    """Tool for resolving conversational references in financial contexts."""
    
    name: str = "reference_resolver"
    description: str = "Resolve conversational references like 'it', 'that year', 'the previous quarter'"
    
    def _run(self, current_question: str, conversation_history: str, extracted_data: str = "") -> str:
        """
        Resolve references in the current question using conversation context.
        
        Args:
            current_question: Question with potential references
            conversation_history: Previous conversation turns
            extracted_data: Previously extracted data for reference
            
        Returns:
            JSON string with resolved references
        """
        try:
            resolution_result = {
                "identified_references": self._identify_references(current_question),
                "resolved_references": self._resolve_references(current_question, conversation_history),
                "confidence_scores": self._calculate_confidence_scores(current_question, conversation_history),
                "contextual_entities": self._extract_contextual_entities(conversation_history),
                "ambiguity_flags": self._flag_ambiguities(current_question, conversation_history)
            }
            
            return json.dumps(resolution_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Reference resolution failed: {str(e)}"})
    
    def _identify_references(self, question: str) -> List[str]:
        """Identify potential references in the question."""
        references = []
        question_lower = question.lower()
        
        reference_patterns = [
            r'\bit\b', r'\bthat\b', r'\bthis\b', r'\bthey\b', r'\bthem\b',
            r'\bprevious\b', r'\blast\b', r'\bprior\b', r'\babove\b', r'\bbelow\b',
            r'\bsame\b', r'\bother\b', r'\bfollowing\b'
        ]
        
        for pattern in reference_patterns:
            matches = re.findall(pattern, question_lower)
            references.extend(matches)
        
        return list(set(references))
    
    def _resolve_references(self, current_question: str, conversation_history: str) -> Dict[str, str]:
        """Resolve identified references using conversation context."""
        resolutions = {}
        
        # Extract entities and values from conversation history
        entities = self._extract_entities_from_history(conversation_history)
        values = self._extract_values_from_history(conversation_history)
        
        question_lower = current_question.lower()
        
        # Resolve common pronouns
        if 'it' in question_lower:
            if entities:
                resolutions['it'] = entities[-1]  # Most recent entity
        
        if 'that' in question_lower:
            if values:
                resolutions['that'] = values[-1]  # Most recent value
        
        if 'previous' in question_lower or 'last' in question_lower:
            temporal_entities = self._extract_temporal_entities(conversation_history)
            if temporal_entities:
                resolutions['previous/last'] = temporal_entities[-1]
        
        return resolutions
    
    def _extract_entities_from_history(self, conversation_history: str) -> List[str]:
        """Extract financial entities from conversation history."""
        financial_terms = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash', 'debt', 'sales', 'costs', 'expenses'
        ]
        
        found_entities = []
        history_lower = conversation_history.lower()
        
        for term in financial_terms:
            if term in history_lower:
                found_entities.append(term)
        
        return found_entities
    
    def _extract_values_from_history(self, conversation_history: str) -> List[str]:
        """Extract numerical values from conversation history."""
        value_pattern = r'[\$£€]?[\d,]+\.?\d*[%BMK]?'
        values = re.findall(value_pattern, conversation_history)
        return values
    
    def _extract_temporal_entities(self, conversation_history: str) -> List[str]:
        """Extract temporal entities (years, quarters) from history."""
        temporal_entities = []
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, conversation_history)
        temporal_entities.extend([f"{y[0]}{y[1]}" for y in years])
        
        # Quarters
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        quarters = re.findall(quarter_pattern, conversation_history, re.IGNORECASE)
        temporal_entities.extend(quarters)
        
        return temporal_entities
    
    def _calculate_confidence_scores(self, current_question: str, conversation_history: str) -> Dict[str, float]:
        """Calculate confidence scores for reference resolutions."""
        confidence_scores = {}
        
        # Simple heuristic: more context = higher confidence
        context_richness = len(conversation_history.split()) / 100.0  # Normalize by word count
        reference_specificity = 1.0 - (current_question.lower().count('it') + 
                                      current_question.lower().count('that')) * 0.2
        
        base_confidence = min(0.9, context_richness * reference_specificity)
        
        confidence_scores['overall'] = base_confidence
        confidence_scores['entity_resolution'] = base_confidence + 0.1
        confidence_scores['temporal_resolution'] = base_confidence
        
        return confidence_scores
    
    def _flag_ambiguities(self, current_question: str, conversation_history: str) -> List[str]:
        """Flag potential ambiguities in reference resolution."""
        ambiguities = []
        
        # Multiple potential referents
        entities = self._extract_entities_from_history(conversation_history)
        if len(entities) > 3 and 'it' in current_question.lower():
            ambiguities.append("Multiple potential referents for 'it'")
        
        # Temporal ambiguity
        temporal_entities = self._extract_temporal_entities(conversation_history)
        if len(temporal_entities) > 2 and any(word in current_question.lower() 
                                             for word in ['previous', 'last']):
            ambiguities.append("Temporal reference ambiguity")
        
        return ambiguities


class TemporalParserTool(BaseTool):
    """Tool for parsing and resolving temporal expressions in financial contexts."""
    
    name: str = "temporal_parser"
    description: str = "Parse temporal expressions and map them to specific time periods in financial data"
    
    def _run(self, question: str, conversation_history: str, document_context: str = "") -> str:
        """
        Parse temporal expressions and resolve them to specific periods.
        
        Args:
            question: Current question with potential temporal expressions
            conversation_history: Previous conversation context
            document_context: Financial documents for temporal anchoring
            
        Returns:
            JSON string with temporal analysis
        """
        try:
            temporal_analysis = {
                "identified_expressions": self._identify_temporal_expressions(question),
                "resolved_periods": self._resolve_temporal_periods(question, conversation_history),
                "document_periods": self._extract_document_periods(document_context),
                "temporal_mapping": self._create_temporal_mapping(question, conversation_history, document_context),
                "sequence_analysis": self._analyse_temporal_sequence(conversation_history)
            }
            
            return json.dumps(temporal_analysis, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Temporal parsing failed: {str(e)}"})
    
    def _identify_temporal_expressions(self, question: str) -> List[str]:
        """Identify temporal expressions in the question."""
        temporal_expressions = []
        
        # Explicit years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, question)
        temporal_expressions.extend([f"{y[0]}{y[1]}" for y in years])
        
        # Quarters
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        quarters = re.findall(quarter_pattern, question, re.IGNORECASE)
        temporal_expressions.extend(quarters)
        
        # Relative expressions
        relative_patterns = [
            r'\bprevious year\b', r'\blast year\b', r'\bnext year\b',
            r'\bprevious quarter\b', r'\blast quarter\b', r'\bnext quarter\b',
            r'\bprior period\b', r'\bcurrent period\b'
        ]
        
        for pattern in relative_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            temporal_expressions.extend(matches)
        
        return temporal_expressions
    
    def _resolve_temporal_periods(self, question: str, conversation_history: str) -> Dict[str, str]:
        """Resolve relative temporal expressions to specific periods."""
        resolutions = {}
        
        # Extract all years and quarters from conversation history
        all_periods = self._extract_all_periods(conversation_history)
        
        question_lower = question.lower()
        
        if 'previous year' in question_lower or 'last year' in question_lower:
            years = [p for p in all_periods if re.match(r'\b(19|20)\d{2}\b', p)]
            if years:
                latest_year = max(years)
                resolutions['previous_year'] = str(int(latest_year) - 1)
        
        if 'previous quarter' in question_lower or 'last quarter' in question_lower:
            quarters = [p for p in all_periods if 'Q' in p]
            if quarters:
                resolutions['previous_quarter'] = self._get_previous_quarter(quarters[-1])
        
        return resolutions
    
    def _extract_document_periods(self, document_context: str) -> List[str]:
        """Extract available time periods from documents."""
        periods = []
        
        if not document_context:
            return periods
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, document_context)
        periods.extend([f"{y[0]}{y[1]}" for y in years])
        
        # Extract quarters
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        quarters = re.findall(quarter_pattern, document_context, re.IGNORECASE)
        periods.extend(quarters)
        
        return list(set(periods))
    
    def _extract_all_periods(self, conversation_history: str) -> List[str]:
        """Extract all temporal periods mentioned in conversation."""
        periods = []
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, conversation_history)
        periods.extend([f"{y[0]}{y[1]}" for y in years])
        
        # Quarters
        quarter_pattern = r'\b(Q[1-4]|quarter [1-4])\b'
        quarters = re.findall(quarter_pattern, conversation_history, re.IGNORECASE)
        periods.extend(quarters)
        
        return list(set(periods))
    
    def _create_temporal_mapping(self, question: str, conversation_history: str, document_context: str) -> Dict[str, Any]:
        """Create mapping between question temporal references and available data periods."""
        mapping = {}
        
        question_periods = self._identify_temporal_expressions(question)
        available_periods = self._extract_document_periods(document_context)
        
        for q_period in question_periods:
            if q_period in available_periods:
                mapping[q_period] = {"status": "exact_match", "mapped_to": q_period}
            else:
                # Try to find closest match
                closest = self._find_closest_period(q_period, available_periods)
                if closest:
                    mapping[q_period] = {"status": "approximate_match", "mapped_to": closest}
                else:
                    mapping[q_period] = {"status": "no_match", "mapped_to": None}
        
        return mapping
    
    def _get_previous_quarter(self, quarter: str) -> str:
        """Get the previous quarter for a given quarter."""
        if 'Q1' in quarter.upper():
            return quarter.replace('Q1', 'Q4').replace('1', '4')  # Previous year Q4
        elif 'Q2' in quarter.upper():
            return quarter.replace('Q2', 'Q1').replace('2', '1')
        elif 'Q3' in quarter.upper():
            return quarter.replace('Q3', 'Q2').replace('3', '2')
        elif 'Q4' in quarter.upper():
            return quarter.replace('Q4', 'Q3').replace('4', '3')
        return quarter
    
    def _find_closest_period(self, target_period: str, available_periods: List[str]) -> Optional[str]:
        """Find the closest matching period from available periods."""
        # Simple heuristic: exact string match or partial match
        for period in available_periods:
            if target_period in period or period in target_period:
                return period
        return None
    
    def _analyse_temporal_sequence(self, conversation_history: str) -> Dict[str, Any]:
        """Analyse the temporal sequence of the conversation."""
        all_periods = self._extract_all_periods(conversation_history)
        
        # Sort periods chronologically (simplified)
        years = sorted([p for p in all_periods if re.match(r'\b(19|20)\d{2}\b', p)])
        quarters = sorted([p for p in all_periods if 'Q' in p])
        
        sequence_analysis = {
            "chronological_years": years,
            "chronological_quarters": quarters,
            "temporal_span": f"{years[0]} to {years[-1]}" if years else "Unknown",
            "conversation_pattern": "sequential" if len(years) > 1 else "single_period"
        }
        
        return sequence_analysis 