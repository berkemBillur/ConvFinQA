"""DSL program execution for ConvFinQA."""

import re
from typing import Union, List, Any
from decimal import Decimal, InvalidOperation


class DSLExecutor:
    """Execute DSL programs from ConvFinQA dataset."""
    
    def __init__(self):
        """Initialise the DSL executor."""
        self.constants = {
            'const_1000': 1000,
            'const_100': 100,
            'const_1': 1,
        }
    
    def execute(self, program: str) -> Union[float, str]:
        """Execute a DSL program.
        
        Args:
            program: DSL program string to execute.
            
        Returns:
            Execution result (number or error message).
        """
        try:
            return self._execute_program(program.strip())
        except Exception as e:
            return f"Execution error: {str(e)}"
    
    def _execute_program(self, program: str) -> Union[float, str]:
        """Internal program execution logic."""
        # Handle simple numeric values
        if self._is_numeric(program):
            return float(program)
        
        # Handle constants
        if program in self.constants:
            return float(self.constants[program])
        
        # Handle comma-separated operations (but not commas inside function calls)
        if ',' in program and not (program.count('(') == 1 and program.count(')') == 1):
            return self._execute_chained_operations(program)
        
        # Handle single operations
        return self._execute_single_operation(program)
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a number."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _execute_chained_operations(self, program: str) -> float:
        """Execute multiple operations separated by commas."""
        operations = self._parse_operations(program)
        results = []
        
        for operation in operations:
            if operation.strip().startswith('#'):
                # Reference to previous result
                ref_index = int(operation.strip()[1:])
                if ref_index < len(results):
                    result = results[ref_index]
                else:
                    raise ValueError(f"Invalid reference: {operation}")
            else:
                # Execute new operation, substituting previous results
                substituted_op = self._substitute_references(operation, results)
                result = self._execute_single_operation(substituted_op)
            
            results.append(result)
        
        return results[-1]  # Return the last result
    
    def _parse_operations(self, program: str) -> List[str]:
        """Parse operations, respecting parentheses."""
        operations = []
        current_op = ""
        paren_count = 0
        
        for char in program:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                operations.append(current_op.strip())
                current_op = ""
                continue
            
            current_op += char
        
        if current_op.strip():
            operations.append(current_op.strip())
        
        return operations
    
    def _substitute_references(self, operation: str, results: List[float]) -> str:
        """Replace #N references with actual values."""
        def replace_ref(match):
            ref_index = int(match.group(1))
            if ref_index < len(results):
                return str(results[ref_index])
            else:
                raise ValueError(f"Invalid reference: {match.group(0)}")
        
        return re.sub(r'#(\d+)', replace_ref, operation)
    
    def _execute_single_operation(self, operation: str) -> float:
        """Execute a single operation."""
        operation = operation.strip()
        
        # Parse operation pattern: function(arg1, arg2, ...)
        match = re.match(r'^(\w+)\((.*)\)$', operation)
        
        if not match:
            # Try as simple value
            if self._is_numeric(operation):
                return float(operation)
            elif operation in self.constants:
                return float(self.constants[operation])
            else:
                raise ValueError(f"Invalid operation format: {operation}")
        
        func_name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        if args_str.strip():
            args = [arg.strip() for arg in args_str.split(',')]
            parsed_args = []
            
            for arg in args:
                if self._is_numeric(arg):
                    parsed_args.append(float(arg))
                elif arg in self.constants:
                    parsed_args.append(float(self.constants[arg]))
                else:
                    raise ValueError(f"Invalid argument: {arg}")
        else:
            parsed_args = []
        
        # Execute the function
        return self._execute_function(func_name, parsed_args)
    
    def _execute_function(self, func_name: str, args: List[float]) -> float:
        """Execute a specific function with arguments."""
        if func_name == 'add':
            if len(args) != 2:
                raise ValueError(f"add() requires 2 arguments, got {len(args)}")
            return args[0] + args[1]
        
        elif func_name == 'subtract':
            if len(args) != 2:
                raise ValueError(f"subtract() requires 2 arguments, got {len(args)}")
            return args[0] - args[1]
        
        elif func_name == 'multiply':
            if len(args) != 2:
                raise ValueError(f"multiply() requires 2 arguments, got {len(args)}")
            return args[0] * args[1]
        
        elif func_name == 'divide':
            if len(args) != 2:
                raise ValueError(f"divide() requires 2 arguments, got {len(args)}")
            
            # Enhanced safe division with gamma safeguard (research-based improvement)
            divisor = args[1]
            EPS = 1e-8  # Gamma safeguard value
            
            if abs(divisor) < EPS:
                # Use gamma safeguard instead of raising error
                safe_divisor = EPS if divisor >= 0 else -EPS
                return args[0] / safe_divisor
            
            return args[0] / divisor
        
        else:
            raise ValueError(f"Unknown function: {func_name}")


def execute_dsl_program(program: str) -> Union[float, str]:
    """Convenience function to execute a DSL program.
    
    Args:
        program: DSL program string.
        
    Returns:
        Execution result or error message.
    """
    executor = DSLExecutor()
    return executor.execute(program) 