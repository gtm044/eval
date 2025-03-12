#!/usr/bin/env python
# examples/prompt_templates_example.py

"""
Example script demonstrating how to use the Jinja templates for prompts.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.prompts import (
    # Default rendered prompts
    synthetic_query_prompt,
    synthetic_valid_answer_prompt,
    synthetic_valid_short_answer_prompt,
    expand_documents_prompt,
    # Functions to render with parameters
    render_synthetic_query_prompt,
    render_synthetic_valid_answer_prompt,
    render_synthetic_valid_short_answer_prompt,
    render_expand_documents_prompt
)

def main():
    """Demonstrate the use of prompt templates."""
    
    print("=" * 80)
    print("DEFAULT PROMPTS (without parameters)")
    print("=" * 80)
    
    print("\nSYNTHETIC QUERY PROMPT:")
    print("-" * 40)
    print(synthetic_query_prompt)
    
    print("\nSYNTHETIC VALID ANSWER PROMPT:")
    print("-" * 40)
    print(synthetic_valid_answer_prompt)
    
    print("\nSYNTHETIC VALID SHORT ANSWER PROMPT:")
    print("-" * 40)
    print(synthetic_valid_short_answer_prompt)
    
    print("\nEXPAND DOCUMENTS PROMPT:")
    print("-" * 40)
    print(expand_documents_prompt)
    
    print("\n" + "=" * 80)
    print("CUSTOMIZED PROMPTS (with parameters)")
    print("=" * 80)
    
    # Example 1: Customize synthetic query prompt
    custom_query_prompt = render_synthetic_query_prompt(
        example_questions=[
            "What is the primary ingredient in chocolate?",
            "How does photosynthesis work?",
            "What year was the first iPhone released?"
        ],
        no_question_response="INSUFFICIENT_INFORMATION"
    )
    
    print("\nCUSTOMIZED SYNTHETIC QUERY PROMPT:")
    print("-" * 40)
    print(custom_query_prompt)
    
    # Example 2: Customize synthetic answer prompt
    custom_answer_prompt = render_synthetic_valid_answer_prompt(
        custom_instructions="Generate a very concise answer that is no more than 15 words long.",
        no_question_marker="INSUFFICIENT_INFORMATION",
        no_answer_response="CANNOT_ANSWER"
    )
    
    print("\nCUSTOMIZED SYNTHETIC VALID ANSWER PROMPT:")
    print("-" * 40)
    print(custom_answer_prompt)
    
    # Example 3: Customize synthetic short answer prompt
    custom_short_answer_prompt = render_synthetic_valid_short_answer_prompt(
        custom_instructions="Generate a single word or phrase as the answer. Do not include any explanations.",
        no_question_marker="INSUFFICIENT_INFORMATION",
        no_answer_response="UNKNOWN"
    )
    
    print("\nCUSTOMIZED SYNTHETIC VALID SHORT ANSWER PROMPT:")
    print("-" * 40)
    print(custom_short_answer_prompt)
    
    # Example 4: Customize expand documents prompt
    custom_expand_prompt = render_expand_documents_prompt(
        custom_instructions="Focus on creating variations that change the tone and style while preserving the exact meaning.",
        limit_specified=True,
        limit=5
    )
    
    print("\nCUSTOMIZED EXPAND DOCUMENTS PROMPT:")
    print("-" * 40)
    print(custom_expand_prompt)

if __name__ == "__main__":
    main() 