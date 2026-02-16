"""Prompt templates for dataloader operations.

This module contains LLM prompt templates used by dataloaders for
text generation tasks such as answer summarization and financial
content analysis.

Prompt Design Principles:
    - Clear task definition with specific instructions
    - Structured output format specification
    - Contextual information injection via format placeholders
    - Emphasis on single-answer output (no conversational fluff)

Usage:
    These prompts are used by framework-specific dataloaders that
    require LLM-based text generation. They are formatted at runtime
    with dataset-specific values.

Example:
        >>> from vectordb.dataloaders.prompts import SUMMARIZE_ANSWERS_PROMPT
        >>> prompt = SUMMARIZE_ANSWERS_PROMPT.format(
        ...     question="What is the capital of France?",
        ...     answers=["Paris", "City of Light", "Capital of France"],
        ... )
"""

# Prompt for consolidating multiple answer aliases into a single canonical answer
# Used by TriviaQA, PopQA, and FactScore dataloaders to normalize answer formats
SUMMARIZE_ANSWERS_PROMPT = """\
Task: Generate the most accurate and relevant answer.

Instructions:
1. Analyze the given question: '{question}'.
2. Review the provided list of answers: {answers}.
3. Craft a response that best addresses the question. The answer can be:
   - A completely new formulation.
   - A refined combination of ideas from the list.

Output: Only provide the final answer, with no additional text or commentary."""
