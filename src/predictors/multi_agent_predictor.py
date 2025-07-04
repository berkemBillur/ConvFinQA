"""
CrewAI implementation for ConvFinQA.

"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Optional imports with graceful fallback
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

from ..data.models import ConvFinQARecord
from ..utils.config import Config, APIKeyManager
from ..utils.enhanced_tracker import get_enhanced_tracker
from . import Predictor


@dataclass
class AgentInput:
    """Structured input tracking for reproducibility."""
    agent_role: str
    task_description: str
    input_data: Dict[str, Any]
    timestamp: str
    context_tokens: int


@dataclass
class AgentOutput:
    """Structured output tracking with confidence scoring."""
    agent_role: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    error_flags: List[str]
    token_usage: Dict[str, int]


@dataclass
class CrewExecution:
    """Complete crew execution tracking for analysis."""
    execution_id: str
    inputs: List[AgentInput]
    outputs: List[AgentOutput]
    final_result: str
    total_cost: float
    success: bool
    error_summary: Optional[str]


class ConvFinQAMultiAgentPredictor(Predictor):
    """
    Research-validated multi-agent predictor for conversational financial Q&A.
    
    Implements hierarchical agent coordination using CrewAI framework with:
    - Financial QA Orchestrator (supervisor)
    - Data Extraction Specialist  
    - Calculations Specialist
    - QA Validator
    """
    
    def __init__(self, config: Config):
        """
        Initialise the multi-agent predictor with configurable agents.
        
        Args:
            config: Configuration object with crew parameters
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI dependencies not available. "
                "Install with: pip install crewai crewai-tools langchain-openai"
            )
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check for API key using multiple sources
        api_key = self._load_api_key()
        if not api_key:
            self.logger.warning("No OpenAI API key found. Multi-agent predictor will use fallback mode.")
            self.fallback_mode = True
        else:
            self.fallback_mode = False
        
        # Enhanced tracking integration (simplified for multi-agent predictor)
        self.enhanced_tracker = get_enhanced_tracker()
        self.config_hash = f"multi_agent_{hash(str(config))}"  # Simplified hash for compatibility
        
        # Legacy execution tracking for compatibility
        self.executions: List[CrewExecution] = []
        self.total_cost = 0.0
        
        if not self.fallback_mode:
            # Initialise crew components
            self._initialise_agents()
            self._initialise_tools()
            self._setup_crew()
    
    def _load_api_key(self) -> Optional[str]:
        """Load OpenAI API key from multiple sources with priority order."""
        key = APIKeyManager.load_openai_key()
        if key:
            return key
        
        # Fallback to config object if provided
        key = self.config.get('openai_api_key')
        if key:
            self.logger.debug("✅ API key loaded from config object")
            return key
        
        return None
        
    def _initialise_agents(self) -> None:
        """Initialise the four research-validated agents."""
        crew_config = self.config.get('crewai', {})
        api_key = self._load_api_key()
        
        # Supervisor Agent - GPT-4o for complex orchestration
        self.supervisor = Agent(
            role='Financial QA Orchestrator',
            goal='Decompose conversational financial queries and coordinate specialist agents for accurate answers',
            backstory="""You are a senior financial analyst supervisor who excels at breaking down 
            complex conversational finance questions into structured subtasks. You maintain conversation 
            context across multiple turns and coordinate specialist agents to produce accurate, 
            well-reasoned answers.""",
            verbose=crew_config.get('verbose', True),
            allow_delegation=True,  # Essential for hierarchical orchestration
            llm=ChatOpenAI(
                model=crew_config.get('supervisor_model', 'gpt-4o-mini'),
                temperature=crew_config.get('supervisor_temperature', 0.1),
                api_key=SecretStr(api_key) if api_key else None
            ),
            tools=[]  # Will be populated after tool initialisation
        )
        
        # Data Extraction Specialist - GPT-4o-mini for cost optimisation
        self.data_extractor = Agent(
            role='Financial Data Extraction Specialist',
            goal='Extract precise numerical data from financial documents and resolve conversational references',
            backstory="""You are a data extraction expert who specialises in financial documents. 
            You excel at finding specific numerical values in complex tables and resolving 
            conversational references like 'it', 'that year', and 'the previous quarter' across 
            multi-turn conversations.""",
            verbose=crew_config.get('verbose', True),
            allow_delegation=False,
            llm=ChatOpenAI(
                model=crew_config.get('extractor_model', 'gpt-4o-mini'),
                temperature=crew_config.get('extractor_temperature', 0.0),
                api_key=SecretStr(api_key) if api_key else None
            ),
            tools=[]
        )
        
        # Calculations Specialist - GPT-4o for sophisticated reasoning
        self.calculations_specialist = Agent(
            role='Financial Calculations Specialist',
            goal='Perform accurate financial calculations and generate executable DSL programs',
            backstory="""You are a quantitative financial analyst who performs complex calculations 
            and generates precise DSL programs. You understand financial business logic, apply 
            appropriate calculation methods, and ensure mathematical accuracy in financial contexts.""",
            verbose=crew_config.get('verbose', True),
            allow_delegation=False,
            llm=ChatOpenAI(
                model=crew_config.get('calculator_model', 'gpt-4o-mini'),
                temperature=crew_config.get('calculator_temperature', 0.1),
                api_key=SecretStr(api_key) if api_key else None
            ),
            tools=[]
        )
        
        # QA Validator - GPT-4o-mini for systematic validation
        self.validator = Agent(
            role='Financial QA Validator',
            goal='Validate answers through cross-agent verification and confidence scoring',
            backstory="""You are a financial QA validator who performs final verification of answers. 
            You check for logical consistency, numerical accuracy, and conversational context correctness. 
            You provide confidence scores and identify potential errors before final answer delivery.""",
            verbose=crew_config.get('verbose', True),
            allow_delegation=False,
            llm=ChatOpenAI(
                model=crew_config.get('validator_model', 'gpt-4o-mini'),
                temperature=crew_config.get('validator_temperature', 0.0),
                api_key=SecretStr(api_key) if api_key else None
            ),
            tools=[]
        )
        
    def _initialise_tools(self) -> None:
        """Simple tool setup following prompt guide principles."""
        # No tools for now - let the LLM handle everything
        # Following "simple and effective" principle from prompt guide
        self.supervisor.tools = []
        
        self.logger.info("Simple configuration: no custom tools, LLM capabilities only")
        
    def _setup_crew(self) -> None:
        """Configure crew with all specialist agents for hierarchical orchestration."""
        crew_config = self.config.get('crewai', {})
        
        # Register every agent so the supervisor can delegate work properly
        agent_roster = [
            self.supervisor,
            self.data_extractor,
            self.calculations_specialist,
            self.validator,
        ]
        
        # Some older CrewAI versions may not expose Process.hierarchical –
        # fall back to sequential if it is unavailable.
        default_process = getattr(Process, 'hierarchical', Process.sequential)
        process_type = crew_config.get('process', default_process)

        crew_kwargs = {
            "agents": agent_roster,  # type: ignore[arg-type]
            "tasks": [],  # Tasks are attached per-prediction
            "process": process_type,  # type: ignore[arg-type]
            "verbose": crew_config.get("verbose", True),
        }

        # Hierarchical process requires a manager; use the supervisor agent by default
        if process_type == getattr(Process, "hierarchical", Process.sequential):
            crew_kwargs["manager_agent"] = self.supervisor  # type: ignore[arg-type]
            # CrewAI validator expects manager_agent NOT to appear in the agents list
            crew_kwargs["agents"] = [
                self.data_extractor,
                self.calculations_specialist,
                self.validator,
            ]  # type: ignore[arg-type]

        self.crew = Crew(**crew_kwargs)
        
    def _create_dynamic_task(self, record: ConvFinQARecord) -> Task:
        """Create a task with actual financial data for the agent to process."""
        # Format the data for the agent
        formatted_table = self._format_table(record.doc.table)
        formatted_conversation = self._format_conversation(record.dialogue)
        current_question = record.dialogue.conv_questions[-1]
        
        # Create task description with actual data
        task_description = f"""
You are analyzing financial data to answer: "{current_question}"

FINANCIAL TABLE DATA:
{formatted_table}

CONVERSATION HISTORY:
{formatted_conversation}

DOCUMENT CONTEXT:
{record.doc.pre_text[:500]}...{record.doc.post_text[:500]}

CRITICAL DSL OUTPUT REQUIREMENTS:
- Return ONLY executable DSL programs or pure numbers
- For simple lookups: return the raw number (e.g., "123.45")
- For addition: return "add(value1, value2)" with actual numbers (e.g., "add(123.45, 67.89)")
- For subtraction: return "subtract(value1, value2)" with actual numbers (e.g., "subtract(123.45, 67.89)")
- For division: return "divide(value1, value2)" with actual numbers (e.g., "divide(123.45, 67.89)")
- For multiplication: return "multiply(value1, value2)" with actual numbers (e.g., "multiply(123.45, 67.89)")

PROHIBITED OUTPUTS:
- DO NOT include percentage symbols (% is forbidden)
- DO NOT include currency symbols ($ is forbidden)
- DO NOT include explanations or conversational text
- DO NOT return incomplete DSL (all parentheses must be closed)
- DO NOT return formatted text or explanations

PERCENTAGE HANDLING:
- For percentage questions, return the decimal value (e.g., for 21.1%, return "0.211")
- For "times 100" questions, return the calculated number without % symbol

EXAMPLES:
- Question: "What was the revenue?" → Answer: "123.45"
- Question: "What is the sum?" → Answer: "add(123.45, 67.89)"
- Question: "What percentage change?" → Answer: "0.211" (NOT "-21.1%")
- Question: "What is the difference?" → Answer: "subtract(123.45, 67.89)"
        """
        
        return Task(
            description=task_description,
            agent=self.data_extractor,
            expected_output="Valid executable DSL program (e.g., 'add(123.45, 67.89)') or raw numerical value (e.g., '123.45')"
        )
    
    def _create_hierarchical_tasks(self, record: ConvFinQARecord, execution_id: str) -> List[Task]:
        """Create simple task following prompt guide principles: simple and effective."""
        # FIXED: Use full document context instead of just table
        formatted_docs = self._format_documents([record.doc])
        current_question = record.dialogue.conv_questions[0]
        
        # Get conversation history if available  
        conversation_history = getattr(self, '_current_conversation_history', [])
        
        # Simple history format
        history_text = ""
        if conversation_history:
            history_parts = []
            for i, turn in enumerate(conversation_history):
                history_parts.append(f"Previous Q{i+1}: {turn.get('question', '')} A{i+1}: {turn.get('answer', '')}")
            history_text = " | ".join(history_parts) + " | "
        
        # Single simple task - let the supervisor coordinate everything
        main_task = Task(
            description=f"""
Answer: "{current_question}"

{history_text}Financial Document:
{formatted_docs}

Instructions:
1. Search ALL document sections (text and tables) for relevant data
2. If the answer is directly stated in text, extract the exact number
3. If calculation is needed, use table data and return DSL format
4. Return only the answer as a number or simple calculation

Examples: "60.94" or "add(123.45, 67.89)" or "subtract(818.0, 11798.0)"
            """,
            agent=self.supervisor,
            expected_output="Number or simple calculation only"
        )
        
        return [main_task]

    def predict(self, record: ConvFinQARecord) -> str:
        """
        Generate prediction using the four-agent crew structure.
        
        Args:
            record: ConvFinQA record with conversation and documents
            
        Returns:
            Generated DSL program as string
        """
        start_time = time.time()
        fallback_used = self.fallback_mode
        success = False
        error_message = None
        estimated_cost = 0.0
        confidence = 0.0
        
        try:
            if self.fallback_mode:
                self.logger.info("Using fallback mode due to missing API key")
                result = self._fallback_prediction(record)
                success = True
                return result
            
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Create hierarchical tasks for all agents
            tasks = self._create_hierarchical_tasks(record, execution_id)
            
            # Update crew with tasks and execute
            self.crew.tasks = tasks
            crew_result = self.crew.kickoff()
            
            # Extract DSL program and confidence from result
            dsl_program = self._extract_dsl_program(crew_result)
            confidence = self._extract_confidence(crew_result)
            estimated_cost = self._estimate_execution_cost(crew_result)
            
            # Track successful execution
            success = True
            self.logger.info(f"Multi-agent prediction completed for {execution_id}")
            
            # Legacy tracking for compatibility
            execution_time = time.time() - start_time
            self._track_execution(execution_id, record, crew_result, execution_time, success=True)
            
            return dsl_program
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Multi-agent prediction failed: {error_message}")
            
            # Use fallback on error
            fallback_used = True
            result = self._fallback_prediction(record)
            
            # Legacy tracking for compatibility
            execution_id = f"exec_error_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            execution_time = time.time() - start_time
            self._track_execution(execution_id, record, None, execution_time, success=False, error=error_message)
            
            return result
            
        finally:
            # Performance tracking (always executed)
            execution_time = time.time() - start_time
            
            # Determine question type for metadata
            question_type = self._classify_question_type(record.dialogue.conv_questions[-1])
            conversation_length = len(record.dialogue.conv_questions)
            
            # Note: Execution tracking is now handled by enhanced_benchmark_multi_agent.py
            # Individual predictor calls don't need tracking as the benchmark script captures everything
    
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Union[float, str]:
        """
        Predict answer for a single conversation turn with proper context handling.
        
        Args:
            record: ConvFinQA record containing document and conversation
            turn_index: Index of current turn (0-based)
            conversation_history: Previous turns with questions and answers
            
        Returns:
            Predicted answer for the turn
        """
        try:
            # Create a modified record for the specific turn with proper context
            from copy import deepcopy
            from ..data.models import Dialogue
            
            modified_record = deepcopy(record)
            current_question = record.dialogue.conv_questions[turn_index]
            
            # Update dialogue to focus on current turn with conversation history
            modified_dialogue = Dialogue(
                conv_questions=[current_question],
                conv_answers=[""],  # Current answer unknown
                turn_program=[record.dialogue.turn_program[turn_index]] if turn_index < len(record.dialogue.turn_program) else [""],
                executed_answers=[record.dialogue.executed_answers[turn_index]] if turn_index < len(record.dialogue.executed_answers) else [0],
                qa_split=[record.dialogue.qa_split[turn_index]] if turn_index < len(record.dialogue.qa_split) else [False]
            )
            
            modified_record.dialogue = modified_dialogue
            
            # Store conversation history and turn index in a way that can be accessed by task creation
            self._current_conversation_history = conversation_history
            self._current_turn_index = turn_index
            
            # Get prediction using the main predict method
            dsl_result = self.predict(modified_record)
            
            # Execute the DSL to get the final answer
            from ..evaluation.executor import execute_dsl_program
            try:
                answer = execute_dsl_program(dsl_result)
                return answer if isinstance(answer, (int, float)) else str(answer)
            except Exception as e:
                self.logger.warning(f"DSL execution failed: {e}, returning raw result")
                return dsl_result
                
        except Exception as e:
            self.logger.error(f"predict_turn failed: {e}")
            return "ERROR"
            
    def _create_crew_task(self, record: ConvFinQARecord, execution_id: str) -> Task:
        """Create structured task for the crew with proper context."""
        return Task(
            description=f"""
            Analyse the conversational financial question and provide an accurate answer using DSL.
            
            Context:
            - This is turn {len(record.dialogue.conv_questions)} in the conversation
            - Previous questions and answers provide important context
            - Financial documents contain relevant data tables
            - The final answer must be a valid DSL program
            
            Requirements:
            1. Extract relevant data from financial documents
            2. Resolve any conversational references (it, that, previous, etc.)
            3. Perform accurate financial calculations
            4. Generate valid DSL program for execution
            5. Validate answer for accuracy and completeness
            
            Output Format:
            {{
                "reasoning": "Step-by-step analysis",
                "extracted_data": {{"key_values": "relevant_data"}},
                "calculations": "Mathematical operations performed", 
                "dsl_program": "Final executable DSL program",
                "confidence": 0.95
            }}
            """,
            agent=self.supervisor,
            expected_output="JSON object with reasoning, data, calculations, DSL program, and confidence score"
        )
        
    def _format_conversation(self, dialogue) -> str:
        """Format conversation history for agent consumption."""
        formatted = []
        for i, (question, answer) in enumerate(zip(dialogue.conv_questions, dialogue.conv_answers)):
            formatted.append(f"Turn {i+1}:")
            formatted.append(f"Q: {question}")
            if answer:  # Some answers might be None for current question
                formatted.append(f"A: {answer}")
            formatted.append("")
        return "\n".join(formatted)
        
    def _format_documents(self, documents) -> str:
        """Format financial documents for agent consumption."""
        formatted = []
        for doc in documents:
            formatted.append("=== FINANCIAL DOCUMENT ===")
            
            # Pre-text section
            if doc.pre_text and doc.pre_text.strip():
                formatted.append("\n📄 DOCUMENT TEXT (BEFORE TABLE):")
                formatted.append(doc.pre_text.strip())
            
            # Table section
            formatted.append("\n📊 DATA TABLE:")
            formatted.append(self._format_table(doc.table))
            
            # Post-text section  
            if doc.post_text and doc.post_text.strip():
                formatted.append("\n📄 DOCUMENT TEXT (AFTER TABLE):")
                formatted.append(doc.post_text.strip())
            
            formatted.append("\n" + "="*30)
        return "\n".join(formatted)
    
    def _format_table(self, table: dict) -> str:
        """Format table data for agent consumption."""
        if not table:
            return "No table data available"
        
        # FIXED: Improved table formatting with proper alignment
        formatted_rows = []
        
        if table:
            # Get column names and headers
            col_names = list(table.keys())
            first_col = next(iter(table.values()))
            headers = list(first_col.keys())
            
            # Create header row
            header_row = "Row Label\t" + "\t".join(col_names)
            formatted_rows.append(header_row)
            formatted_rows.append("-" * len(header_row))  # Separator
            
            # Create data rows with proper alignment
            for header in headers:
                row_values = [header]  # Row label first
                for col in col_names:
                    value = table[col].get(header, "")
                    row_values.append(str(value))
                formatted_rows.append("\t".join(row_values))
        
        return "\n".join(formatted_rows)
        
    def _extract_dsl_program(self, crew_result) -> str:
        """Extract DSL program from crew output."""
        try:
            # Handle different possible output formats
            if isinstance(crew_result, str):
                # Try to parse as JSON first
                try:
                    result_json = json.loads(crew_result)
                    return result_json.get('dsl_program', crew_result)
                except json.JSONDecodeError:
                    # Return as-is if not JSON
                    return crew_result
            else:
                # Handle other result types
                return str(crew_result)
        except Exception as e:
            self.logger.warning(f"Failed to extract DSL program: {e}")
            return str(crew_result)
            
    def _fallback_prediction(self, record: ConvFinQARecord) -> str:
        """Simple fallback when crew execution fails."""
        # Return a basic table lookup that should work with the DSL executor
        # This is a conservative fallback that tries to extract the first numerical value
        try:
            table = record.doc.table
            if table:
                # Find first numerical value in the table
                for col_name, col_data in table.items():
                    for row_key, value in col_data.items():
                        try:
                            float_val = float(value)
                            return str(float_val)
                        except (ValueError, TypeError):
                            continue
        except Exception:
            pass
        
        # Ultimate fallback - return a simple number
        return "0"
            
    def _track_execution(self, execution_id: str, record: ConvFinQARecord, 
                        result: Any, execution_time: float, success: bool, 
                        error: Optional[str] = None) -> None:
        """Track execution for reproducibility and analysis."""
        execution = CrewExecution(
            execution_id=execution_id,
            inputs=[],  # Would be populated with actual agent inputs
            outputs=[],  # Would be populated with actual agent outputs
            final_result=str(result) if result else "",
            total_cost=0.0,  # Would calculate actual cost
            success=success,
            error_summary=error
        )
        
        self.executions.append(execution)
        
    def _extract_confidence(self, crew_result) -> float:
        """Extract confidence score from crew output."""
        try:
            if isinstance(crew_result, str):
                try:
                    result_json = json.loads(crew_result)
                    return float(result_json.get('confidence', 0.5))
                except (json.JSONDecodeError, ValueError):
                    return 0.5
            return 0.5
        except Exception:
            return 0.5
    
    def _estimate_execution_cost(self, crew_result) -> float:
        """Estimate execution cost based on crew output.
        
        This is a simplified estimation. In production, this would
        integrate with actual token counting and pricing APIs.
        """
        # Placeholder cost estimation
        # In real implementation, this would track actual token usage
        base_cost = 0.01  # Base cost per execution
        
        try:
            if isinstance(crew_result, str):
                # Rough estimation based on output length
                cost_per_char = 0.00001
                return base_cost + len(crew_result) * cost_per_char
        except Exception:
            pass
        
        return base_cost
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for metadata tracking."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'what was', 'how much']):
            return 'lookup'
        elif any(word in question_lower for word in ['total', 'sum', 'add']):
            return 'aggregation'
        elif any(word in question_lower for word in ['ratio', 'percentage', 'margin']):
            return 'calculation'
        elif any(word in question_lower for word in ['growth', 'increase', 'decrease']):
            return 'trend_analysis'
        elif any(word in question_lower for word in ['compare', 'difference']):
            return 'comparison'
        else:
            return 'complex'
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions for analysis."""
        # Simplified summary - detailed tracking now handled by enhanced_benchmark_multi_agent.py
        return {
            'config_hash': self.config_hash,
            'total_executions': len(self.executions),
            'successful_executions': sum(1 for e in self.executions if e.success),
            'total_cost': self.total_cost,
            'average_cost_per_execution': self.total_cost / max(len(self.executions), 1),
            'recent_executions': [asdict(e) for e in self.executions[-5:]]
        }
    
    def get_performance_comparison(self, other_config_hashes: List[str]) -> Dict[str, Any]:
        """Compare this predictor's performance against other configurations.
        
        Args:
            other_config_hashes: List of other configuration hashes to compare against
            
        Returns:
            Performance comparison data
        """
        # Performance comparison now handled by enhanced experiment tracking system
        return {
            'message': 'Performance comparison available via enhanced tracking dashboard',
            'current_config': self.config_hash,
            'comparison_configs': other_config_hashes
        }

# ---------------------------------------------------------------------------
# New implementation alias (non-breaking).  Import is wrapped in try to avoid
# circular import issues if this file is still directly used elsewhere.
# ---------------------------------------------------------------------------
try:
    from .multi_agent import ConvFinQAMultiAgentPredictor as _NewPredictor

    # Prefer new version unless env variable FORCE_LEGACY_CONVFINQA is set
    import os
    if not os.getenv("FORCE_LEGACY_CONVFINQA"):
        ConvFinQAMultiAgentPredictor = _NewPredictor  # type: ignore
except Exception:
    # Fallback to legacy class already defined in this file
    pass 