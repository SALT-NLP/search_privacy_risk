"""
This module contains all the prompts used in the application.
Each prompt is defined as a function that takes the necessary parameters and returns the formatted prompt string.
"""

def get_app_instruction_prompt(api_spec: str, memory: str, instruction: str) -> str:
    """
    Generate prompt for processing user instruction on apps.
    
    Args:
        api_spec: API specification of available apps
        memory: Current agent memory
        instruction: User instruction to process
        
    Returns:
        Formatted prompt string
    """
    api_spec = api_spec.strip()
    memory = memory.strip()
    instruction = instruction.strip()

    return f"""# Agent System Prompt

You are an autonomous agent representing a user across applications and take actions on behalf of the user.

## API Specifications

{api_spec}

## Agent Memory

{memory}

## User Instruction

{instruction}

## Task

Can you predict the app, function, and parameters from the user instruction?

## Reasoning Process

Before providing your answer, please think step by step.

## Response Format

Return your response as a JSON object with the following structure:

```json
{{
  "reasoning": [
    "What are the available options based on the API?",
    "What would be the most appropriate action?"
  ],
  "function_call": {{
    "app": "app_name",
    "function": "function_name",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
}}
```
"""

def get_agent_instruction_prompt(api_spec: str, memory: str, instruction: str) -> str:
    """
    Generate prompt for processing user instruction on agent.
    
    Args:
        api_spec: API specification of the agent
        memory: Current agent memory
        instruction: User instruction to process
        
    Returns:
        Formatted prompt string
    """
    api_spec = api_spec.strip()
    memory = memory.strip()
    instruction = instruction.strip()

    return f"""# Agent System Prompt

You are an autonomous agent representing a user across applications and take actions on behalf of the user.

## API Specifications

{api_spec}

## Agent Memory

{memory}

## User Instruction

{instruction}

## Task

Can you predict the function from the user instruction?

## Reasoning Process

Before providing your answer, please think step by step.

## Response Format

Return your response as a JSON object with the following structure:

```json
{{
  "reasoning": [
    "What are the available options based on the API?",
    "What would be the most appropriate action?"
  ],
  "function_call": {{
    "agent": "agent_name",
    "function": "function_name",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
}}
```
"""