"""Pydantic models for ConvFinQA dataset structures."""

from pydantic import BaseModel, Field
from typing import Union


class Features(BaseModel):
    """Features of a ConvFinQA record, created by Tomoro to help understand the data."""
    
    num_dialogue_turns: int = Field(
        description="The number of turns in the dialogue, calculated from the length of conv_questions"
    )
    has_type2_question: bool = Field(
        description="Whether the dialogue has a type 2 question, calculated if qa_split contains a 1 this will return true"
    )
    has_duplicate_columns: bool = Field(
        description="Whether the table has duplicate column names not fully addressed during cleaning. We suffix the duplicate column headers with a number if there was no algorithmic fix. e.g. 'Revenue (1)' or 'Revenue (2)'"
    )
    has_non_numeric_values: bool = Field(
        description="Whether the table has non-numeric values"
    )


class Document(BaseModel):
    """Document containing financial information with text and tables."""
    
    pre_text: str = Field(description="The text before the table in the document")
    post_text: str = Field(description="The text after the table in the document")
    table: dict[str, dict[str, Union[float, str, int]]] = Field(
        description="The table of the document as a dictionary"
    )


class Dialogue(BaseModel):
    """Conversational dialogue with questions, answers, and programs."""
    
    conv_questions: list[str] = Field(
        description="The questions in the conversation dialogue, originally called 'dialogue_break'"
    )
    conv_answers: list[str] = Field(
        description="The answers to each question turn, derived from 'answer_list' and original FinQA answers"
    )
    turn_program: list[str] = Field(
        description="The DSL turn program for each question turn"
    )
    executed_answers: list[Union[float, str]] = Field(
        description="The golden program execution results for each question turn"
    )
    qa_split: list[bool] = Field(
        description="This field indicates the source of each question turn - 0 if from the decomposition of the first FinQA question, 1 if from the second. For the Type I simple conversations, this field is all 0s."
    )


class ConvFinQARecord(BaseModel):
    """Complete ConvFinQA record containing document, dialogue, and metadata."""
    
    id: str = Field(description="The id of the record")
    doc: Document = Field(description="The document")
    dialogue: Dialogue = Field(description="The conversational dialogue")
    features: Features = Field(description="The features of the record, created by Tomoro to help understand the data") 