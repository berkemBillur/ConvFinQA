"""
Validation tools for ConvFinQA CrewAI implementation.

Provides cross-agent verification, confidence scoring, and error detection tools.
"""

import json
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool


class CrossValidationTool(BaseTool):
    """Tool for cross-agent verification of outputs."""
    
    name: str = "cross_validator"
    description: str = "Perform cross-agent verification of financial analysis outputs"
    
    def _run(self, extraction_result: str, calculation_result: str, dsl_program: str) -> str:
        """
        Cross-validate outputs from different agents.
        
        Args:
            extraction_result: Data extraction agent output
            calculation_result: Calculation agent output  
            dsl_program: Generated DSL program
            
        Returns:
            JSON string with validation results
        """
        try:
            # Parse inputs
            extraction_data = json.loads(extraction_result) if isinstance(extraction_result, str) else extraction_result
            calculation_data = json.loads(calculation_result) if isinstance(calculation_result, str) else calculation_result
            
            validation_result = {
                "data_consistency": self._check_data_consistency(extraction_data, calculation_data),
                "logic_consistency": self._check_logic_consistency(calculation_data, dsl_program),
                "completeness_check": self._check_completeness(extraction_data, calculation_data),
                "cross_agent_confidence": self._calculate_cross_confidence(extraction_data, calculation_data),
                "identified_discrepancies": self._identify_discrepancies(extraction_data, calculation_data),
                "overall_validation": "passed"
            }
            
            # Determine overall validation status
            if validation_result["identified_discrepancies"]:
                validation_result["overall_validation"] = "failed"
            elif len(validation_result["data_consistency"]) > 0:
                validation_result["overall_validation"] = "warning"
            
            return json.dumps(validation_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Cross-validation failed: {str(e)}", "overall_validation": "error"})
    
    def _check_data_consistency(self, extraction_data: Dict[str, Any], calculation_data: Dict[str, Any]) -> List[str]:
        """Check consistency between extracted data and calculation inputs."""
        inconsistencies = []
        
        # Check if calculation inputs match extracted values
        calc_inputs = calculation_data.get('input_values', {})
        extracted_values = extraction_data.get('extracted_values', {})
        
        for key, value in calc_inputs.items():
            if key in extracted_values:
                if str(value) != str(extracted_values[key]):
                    inconsistencies.append(f"Mismatch in {key}: extracted {extracted_values[key]}, used {value}")
        
        return inconsistencies
    
    def _check_logic_consistency(self, calculation_data: Dict[str, Any], dsl_program: str) -> List[str]:
        """Check consistency between calculation logic and DSL program."""
        inconsistencies = []
        
        calc_type = calculation_data.get('calculation_type', '').lower()
        dsl_lower = dsl_program.lower()
        
        # Basic consistency checks
        if "ratio" in calc_type and "divide" not in dsl_lower:
            inconsistencies.append("Ratio calculation should use divide operation in DSL")
        elif "sum" in calc_type and "add" not in dsl_lower:
            inconsistencies.append("Sum calculation should use add operation in DSL")
        elif "difference" in calc_type and "subtract" not in dsl_lower:
            inconsistencies.append("Difference calculation should use subtract operation in DSL")
        
        return inconsistencies
    
    def _check_completeness(self, extraction_data: Dict[str, Any], calculation_data: Dict[str, Any]) -> Dict[str, bool]:
        """Check completeness of data and calculations."""
        completeness = {
            "has_extracted_data": bool(extraction_data.get('extracted_values')),
            "has_calculation_result": bool(calculation_data.get('result')),
            "has_validation_checks": bool(calculation_data.get('validation_checks')),
            "data_quality_assessed": bool(extraction_data.get('data_quality'))
        }
        
        return completeness
    
    def _calculate_cross_confidence(self, extraction_data: Dict[str, Any], calculation_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on cross-agent consistency."""
        base_confidence = 0.8
        
        # Reduce confidence for inconsistencies
        extraction_quality = extraction_data.get('data_quality', {}).get('data_completeness', 'medium')
        validation_errors = len(calculation_data.get('validation_checks', []))
        
        if extraction_quality == 'low':
            base_confidence -= 0.2
        elif extraction_quality == 'medium':
            base_confidence -= 0.1
        
        if validation_errors > 0:
            base_confidence -= min(0.3, validation_errors * 0.1)
        
        return max(0.1, base_confidence)
    
    def _identify_discrepancies(self, extraction_data: Dict[str, Any], calculation_data: Dict[str, Any]) -> List[str]:
        """Identify major discrepancies requiring attention."""
        discrepancies = []
        
        # Check for major data issues
        if not extraction_data.get('extracted_values'):
            discrepancies.append("No data extracted - cannot proceed with calculation")
        
        if calculation_data.get('result') is None:
            discrepancies.append("No calculation result produced")
        
        validation_errors = calculation_data.get('validation_checks', [])
        critical_errors = [error for error in validation_errors if 'zero' in error.lower() or 'invalid' in error.lower()]
        
        if critical_errors:
            discrepancies.extend(critical_errors)
        
        return discrepancies


class ConfidenceScorerTool(BaseTool):
    """Tool for calculating confidence scores for financial QA outputs."""
    
    name: str = "confidence_scorer"
    description: str = "Calculate confidence scores for financial analysis outputs based on multiple factors"
    
    def _run(self, agent_outputs: str, question_complexity: str = "medium") -> str:
        """
        Calculate comprehensive confidence scores.
        
        Args:
            agent_outputs: JSON string with all agent outputs
            question_complexity: Assessed complexity of the question
            
        Returns:
            JSON string with confidence analysis
        """
        try:
            # Parse agent outputs
            outputs = json.loads(agent_outputs) if isinstance(agent_outputs, str) else agent_outputs
            
            confidence_analysis = {
                "individual_confidences": self._calculate_individual_confidences(outputs),
                "cross_validation_confidence": self._calculate_cross_validation_confidence(outputs),
                "data_quality_confidence": self._calculate_data_quality_confidence(outputs),
                "calculation_confidence": self._calculate_calculation_confidence(outputs),
                "overall_confidence": 0.0,
                "confidence_factors": self._identify_confidence_factors(outputs),
                "risk_assessment": self._assess_risk_factors(outputs, question_complexity)
            }
            
            # Calculate overall confidence
            confidence_analysis["overall_confidence"] = self._calculate_overall_confidence(confidence_analysis)
            
            return json.dumps(confidence_analysis, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Confidence scoring failed: {str(e)}", "overall_confidence": 0.1})
    
    def _calculate_individual_confidences(self, outputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence for each agent's output."""
        individual_confidences = {}
        
        # Data extraction confidence
        extraction_data = outputs.get('extraction', {})
        data_quality = extraction_data.get('data_quality', {})
        completeness = data_quality.get('data_completeness', 'medium')
        
        if completeness == 'high':
            individual_confidences['extraction'] = 0.9
        elif completeness == 'medium':
            individual_confidences['extraction'] = 0.7
        else:
            individual_confidences['extraction'] = 0.4
        
        # Calculation confidence
        calculation_data = outputs.get('calculation', {})
        validation_errors = len(calculation_data.get('validation_checks', []))
        
        if validation_errors == 0:
            individual_confidences['calculation'] = 0.9
        elif validation_errors <= 2:
            individual_confidences['calculation'] = 0.6
        else:
            individual_confidences['calculation'] = 0.3
        
        return individual_confidences
    
    def _calculate_cross_validation_confidence(self, outputs: Dict[str, Any]) -> float:
        """Calculate confidence based on cross-validation results."""
        cross_validation = outputs.get('cross_validation', {})
        
        if cross_validation.get('overall_validation') == 'passed':
            return 0.9
        elif cross_validation.get('overall_validation') == 'warning':
            return 0.6
        else:
            return 0.2
    
    def _calculate_data_quality_confidence(self, outputs: Dict[str, Any]) -> float:
        """Calculate confidence based on data quality metrics."""
        extraction_data = outputs.get('extraction', {})
        data_quality = extraction_data.get('data_quality', {})
        
        missing_values = data_quality.get('missing_values_indicators', 0)
        total_values = data_quality.get('numerical_values_found', 1)
        
        missing_ratio = missing_values / max(total_values, 1)
        
        if missing_ratio < 0.1:
            return 0.9
        elif missing_ratio < 0.3:
            return 0.6
        else:
            return 0.3
    
    def _calculate_calculation_confidence(self, outputs: Dict[str, Any]) -> float:
        """Calculate confidence based on calculation quality."""
        calculation_data = outputs.get('calculation', {})
        
        # Check if result is reasonable
        result = calculation_data.get('result', 0)
        calc_type = calculation_data.get('calculation_type', '').lower()
        
        if 'ratio' in calc_type and (result < 0 or result > 10):
            return 0.4  # Unusual ratio values
        elif 'growth' in calc_type and abs(result) > 500:
            return 0.3  # Extreme growth rates
        else:
            return 0.8  # Normal ranges
    
    def _calculate_overall_confidence(self, confidence_analysis: Dict[str, Any]) -> float:
        """Calculate weighted overall confidence score."""
        individual = confidence_analysis.get('individual_confidences', {})
        cross_val = confidence_analysis.get('cross_validation_confidence', 0.5)
        data_quality = confidence_analysis.get('data_quality_confidence', 0.5)
        calculation = confidence_analysis.get('calculation_confidence', 0.5)
        
        # Weighted average
        weights = {
            'extraction': 0.25,
            'calculation': 0.25,
            'cross_validation': 0.25,
            'data_quality': 0.25
        }
        
        overall = (
            individual.get('extraction', 0.5) * weights['extraction'] +
            individual.get('calculation', 0.5) * weights['calculation'] +
            cross_val * weights['cross_validation'] +
            data_quality * weights['data_quality']
        )
        
        return round(overall, 3)
    
    def _identify_confidence_factors(self, outputs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify factors affecting confidence."""
        factors = {
            "positive_factors": [],
            "negative_factors": []
        }
        
        # Check for positive factors
        extraction_data = outputs.get('extraction', {})
        if extraction_data.get('data_quality', {}).get('data_completeness') == 'high':
            factors["positive_factors"].append("High data completeness")
        
        cross_validation = outputs.get('cross_validation', {})
        if cross_validation.get('overall_validation') == 'passed':
            factors["positive_factors"].append("Cross-validation passed")
        
        # Check for negative factors
        if extraction_data.get('data_quality', {}).get('missing_values_indicators', 0) > 5:
            factors["negative_factors"].append("High number of missing values")
        
        calculation_data = outputs.get('calculation', {})
        if calculation_data.get('validation_checks'):
            factors["negative_factors"].append("Calculation validation warnings")
        
        return factors
    
    def _assess_risk_factors(self, outputs: Dict[str, Any], question_complexity: str) -> Dict[str, Any]:
        """Assess risk factors for the analysis."""
        risk_assessment = {
            "complexity_risk": question_complexity,
            "data_risk": "low",
            "calculation_risk": "low",
            "overall_risk": "low"
        }
        
        # Assess data risk
        extraction_data = outputs.get('extraction', {})
        missing_ratio = extraction_data.get('data_quality', {}).get('missing_values_indicators', 0) / max(
            extraction_data.get('data_quality', {}).get('numerical_values_found', 1), 1)
        
        if missing_ratio > 0.3:
            risk_assessment["data_risk"] = "high"
        elif missing_ratio > 0.1:
            risk_assessment["data_risk"] = "medium"
        
        # Assess calculation risk
        calculation_data = outputs.get('calculation', {})
        if len(calculation_data.get('validation_checks', [])) > 2:
            risk_assessment["calculation_risk"] = "high"
        elif len(calculation_data.get('validation_checks', [])) > 0:
            risk_assessment["calculation_risk"] = "medium"
        
        # Overall risk
        risk_factors = [risk_assessment[key] for key in ["complexity_risk", "data_risk", "calculation_risk"]]
        if "high" in risk_factors:
            risk_assessment["overall_risk"] = "high"
        elif "medium" in risk_factors:
            risk_assessment["overall_risk"] = "medium"
        
        return risk_assessment


class ErrorDetectorTool(BaseTool):
    """Tool for detecting common errors and patterns in financial QA."""
    
    name: str = "error_detector"
    description: str = "Detect common error patterns and potential issues in financial analysis"
    
    def _run(self, complete_analysis: str, question: str) -> str:
        """
        Detect errors and potential issues in the complete analysis.
        
        Args:
            complete_analysis: Complete analysis from all agents
            question: Original question for context
            
        Returns:
            JSON string with error detection results
        """
        try:
            # Parse complete analysis
            analysis = json.loads(complete_analysis) if isinstance(complete_analysis, str) else complete_analysis
            
            error_detection = {
                "detected_errors": self._detect_errors(analysis, question),
                "warning_patterns": self._detect_warning_patterns(analysis),
                "data_anomalies": self._detect_data_anomalies(analysis),
                "logic_issues": self._detect_logic_issues(analysis, question),
                "recommendations": self._generate_recommendations(analysis),
                "error_severity": "low"
            }
            
            # Assess overall error severity
            error_detection["error_severity"] = self._assess_error_severity(error_detection)
            
            return json.dumps(error_detection, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error detection failed: {str(e)}", "error_severity": "high"})
    
    def _detect_errors(self, analysis: Dict[str, Any], question: str) -> List[str]:
        """Detect clear errors in the analysis."""
        errors = []
        
        # Check for division by zero
        calculation_data = analysis.get('calculation', {})
        if 'zero' in str(calculation_data.get('validation_checks', [])).lower():
            errors.append("Division by zero detected in calculations")
        
        # Check for missing critical data
        extraction_data = analysis.get('extraction', {})
        if not extraction_data.get('extracted_values'):
            errors.append("No financial data extracted from documents")
        
        # Check for unrealistic results
        result = calculation_data.get('result', 0)
        if isinstance(result, (int, float)) and abs(result) > 1e6:
            errors.append("Calculation result appears unrealistically large")
        
        return errors
    
    def _detect_warning_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """Detect patterns that warrant warnings."""
        warnings = []
        
        # Check confidence levels
        confidence_data = analysis.get('confidence', {})
        overall_confidence = confidence_data.get('overall_confidence', 0.5)
        
        if overall_confidence < 0.4:
            warnings.append("Low overall confidence in analysis")
        
        # Check for data quality issues
        extraction_data = analysis.get('extraction', {})
        data_quality = extraction_data.get('data_quality', {})
        
        if data_quality.get('data_completeness') == 'low':
            warnings.append("Low data completeness may affect accuracy")
        
        return warnings
    
    def _detect_data_anomalies(self, analysis: Dict[str, Any]) -> List[str]:
        """Detect anomalies in the financial data."""
        anomalies = []
        
        extraction_data = analysis.get('extraction', {})
        extracted_values = extraction_data.get('extracted_values', {})
        
        # Check for negative values where unexpected
        for key, values in extracted_values.items():
            if isinstance(values, list):
                for value in values:
                    try:
                        num_value = float(value.replace('$', '').replace(',', ''))
                        if num_value < 0 and 'revenue' in key.lower():
                            anomalies.append(f"Negative revenue value detected: {value}")
                    except (ValueError, AttributeError):
                        pass
        
        return anomalies
    
    def _detect_logic_issues(self, analysis: Dict[str, Any], question: str) -> List[str]:
        """Detect logical inconsistencies."""
        logic_issues = []
        
        # Check question-answer alignment
        question_lower = question.lower()
        calculation_data = analysis.get('calculation', {})
        calc_type = calculation_data.get('calculation_type', '').lower()
        
        if 'ratio' in question_lower and 'ratio' not in calc_type:
            logic_issues.append("Question asks for ratio but calculation type doesn't match")
        
        if 'growth' in question_lower and 'growth' not in calc_type:
            logic_issues.append("Question asks for growth but calculation type doesn't match")
        
        return logic_issues
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving the analysis."""
        recommendations = []
        
        # Based on confidence levels
        confidence_data = analysis.get('confidence', {})
        if confidence_data.get('overall_confidence', 0.5) < 0.6:
            recommendations.append("Consider requesting additional data validation")
        
        # Based on data quality
        extraction_data = analysis.get('extraction', {})
        if extraction_data.get('data_quality', {}).get('missing_values_indicators', 0) > 3:
            recommendations.append("Review source documents for data completeness")
        
        return recommendations
    
    def _assess_error_severity(self, error_detection: Dict[str, Any]) -> str:
        """Assess the overall severity of detected errors."""
        errors = error_detection.get('detected_errors', [])
        warnings = error_detection.get('warning_patterns', [])
        anomalies = error_detection.get('data_anomalies', [])
        
        if len(errors) > 2:
            return "high"
        elif len(errors) > 0 or len(anomalies) > 2:
            return "medium"
        elif len(warnings) > 0:
            return "low"
        else:
            return "none" 