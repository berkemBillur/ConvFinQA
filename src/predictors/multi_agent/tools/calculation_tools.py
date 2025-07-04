"""
Calculation tools for ConvFinQA CrewAI implementation.

Provides financial calculation, DSL generation, and validation tools.
"""

import json
import re
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool


class CalculationTool(BaseTool):
    """Tool for performing financial calculations."""
    
    name: str = "financial_calculator"
    description: str = "Perform accurate financial calculations based on extracted data"
    
    def _run(self, calculation_type: str, data_values: str, context: str = "") -> str:
        """
        Perform financial calculations.
        
        Args:
            calculation_type: Type of calculation needed
            data_values: Extracted numerical data as JSON
            context: Additional context for calculation
            
        Returns:
            JSON string with calculation results
        """
        try:
            # Parse input data
            values = json.loads(data_values) if isinstance(data_values, str) else data_values
            
            calculation_result = {
                "calculation_type": calculation_type,
                "input_values": values,
                "result": self._perform_calculation(calculation_type, values),
                "formula_used": self._get_formula(calculation_type),
                "validation_checks": self._validate_calculation(calculation_type, values)
            }
            
            return json.dumps(calculation_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Calculation failed: {str(e)}"})
    
    def _perform_calculation(self, calc_type: str, values: Dict[str, Any]) -> float:
        """Perform the actual calculation based on type."""
        calc_type_lower = calc_type.lower()
        
        if "ratio" in calc_type_lower:
            numerator = float(values.get('numerator', 0))
            denominator = float(values.get('denominator', 1))
            return numerator / denominator if denominator != 0 else 0
            
        elif "growth" in calc_type_lower:
            current = float(values.get('current_value', 0))
            previous = float(values.get('previous_value', 1))
            return ((current - previous) / previous) * 100 if previous != 0 else 0
            
        elif "sum" in calc_type_lower or "total" in calc_type_lower:
            return sum(float(v) for v in values.values() if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit())
            
        elif "difference" in calc_type_lower:
            value1 = float(values.get('value1', 0))
            value2 = float(values.get('value2', 0))
            return value1 - value2
            
        else:
            # Default to sum for unknown calculation types
            return sum(float(v) for v in values.values() if isinstance(v, (int, float, str)) and str(v).replace('.', '').isdigit())
    
    def _get_formula(self, calc_type: str) -> str:
        """Get the formula used for the calculation."""
        calc_type_lower = calc_type.lower()
        
        formulas = {
            "ratio": "numerator / denominator",
            "growth": "((current - previous) / previous) * 100",
            "sum": "sum(all_values)",
            "difference": "value1 - value2",
            "margin": "(revenue - costs) / revenue * 100"
        }
        
        for key, formula in formulas.items():
            if key in calc_type_lower:
                return formula
        
        return "sum(values)"
    
    def _validate_calculation(self, calc_type: str, values: Dict[str, Any]) -> List[str]:
        """Validate the calculation inputs and results."""
        validation_checks = []
        
        # Check for missing values
        if not values:
            validation_checks.append("No input values provided")
            
        # Check for zero denominators in ratios
        if "ratio" in calc_type.lower() and values.get('denominator', 1) == 0:
            validation_checks.append("Division by zero in ratio calculation")
            
        # Check for negative values in growth calculations
        if "growth" in calc_type.lower() and values.get('previous_value', 1) <= 0:
            validation_checks.append("Invalid base value for growth calculation")
            
        return validation_checks


class DSLGeneratorTool(BaseTool):
    """Tool for generating executable DSL programs."""
    
    name: str = "dsl_generator"
    description: str = "Generate executable DSL programs for financial calculations"
    
    def _run(self, calculation_result: str, question_context: str) -> str:
        """
        Generate DSL program based on calculation results.
        
        Args:
            calculation_result: JSON string with calculation details
            question_context: Original question context
            
        Returns:
            Executable DSL program string
        """
        try:
            # Parse calculation result
            calc_data = json.loads(calculation_result) if isinstance(calculation_result, str) else calculation_result
            
            dsl_program = self._build_dsl_program(calc_data, question_context)
            
            return dsl_program
            
        except Exception as e:
            return f"# Error generating DSL: {str(e)}\ntable_lookup(table, value)"
    
    def _build_dsl_program(self, calc_data: Dict[str, Any], context: str) -> str:
        """Build the actual DSL program."""
        calc_type = calc_data.get('calculation_type', '').lower()
        
        if "ratio" in calc_type:
            return "divide(numerator_value, denominator_value)"
            
        elif "growth" in calc_type:
            return "subtract(current_value, previous_value) / previous_value * 100"
            
        elif "sum" in calc_type or "total" in calc_type:
            return "add(value1, value2)"
            
        elif "difference" in calc_type:
            return "subtract(value1, value2)"
            
        else:
            # Default lookup
            return "table_lookup(financial_table, lookup_key)"


class FinancialValidatorTool(BaseTool):
    """Tool for validating financial calculations and business logic."""
    
    name: str = "financial_validator"
    description: str = "Validate financial calculations for business logic correctness"
    
    def _run(self, calculation_result: str, business_context: str = "") -> str:
        """
        Validate financial calculation for business logic.
        
        Args:
            calculation_result: Calculation result to validate
            business_context: Business context for validation
            
        Returns:
            JSON string with validation results
        """
        try:
            # Parse calculation result
            calc_data = json.loads(calculation_result) if isinstance(calculation_result, str) else calculation_result
            
            validation_result = {
                "is_valid": True,
                "business_logic_checks": self._check_business_logic(calc_data),
                "range_validation": self._validate_ranges(calc_data),
                "consistency_checks": self._check_consistency(calc_data),
                "warnings": [],
                "recommendations": []
            }
            
            # Determine overall validity
            if validation_result["business_logic_checks"] or validation_result["range_validation"]:
                validation_result["is_valid"] = False
            
            return json.dumps(validation_result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Validation failed: {str(e)}", "is_valid": False})
    
    def _check_business_logic(self, calc_data: Dict[str, Any]) -> List[str]:
        """Check business logic constraints."""
        errors = []
        calc_type = calc_data.get('calculation_type', '').lower()
        result = calc_data.get('result', 0)
        
        # Check for unrealistic ratios
        if "ratio" in calc_type and result > 100:
            errors.append("Ratio exceeds reasonable business limits (>100)")
            
        # Check for extreme growth rates
        if "growth" in calc_type and abs(result) > 1000:
            errors.append("Growth rate seems unrealistic (>1000%)")
            
        return errors
    
    def _validate_ranges(self, calc_data: Dict[str, Any]) -> List[str]:
        """Validate that values are within expected ranges."""
        errors = []
        result = calc_data.get('result', 0)
        
        # Check for negative values where inappropriate
        calc_type = calc_data.get('calculation_type', '').lower()
        if "ratio" in calc_type and result < 0:
            errors.append("Negative ratio may indicate data extraction error")
            
        return errors
    
    def _check_consistency(self, calc_data: Dict[str, Any]) -> List[str]:
        """Check internal consistency of calculation."""
        warnings = []
        
        # This would include more sophisticated consistency checks
        # For now, just basic validation
        
        return warnings 