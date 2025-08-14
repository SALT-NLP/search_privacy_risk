from datetime import datetime
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_multi_turn_system_message(memory: str, long_term_goal: str, start_time: str, time_limit: str) -> str:
    return f"""# Agent System Context

You are an autonomous agent representing a user across multiple tools.

## Agent System Architecture

The agent operates using:
1. **Tasks** - Long-running activities with specific goals
2. **Triggers** - Events that initiate action cycles
3. **Action Cycles** - Short, self-contained sequences of actions (tool calls) responding to triggers
4. **Memory** - Information about your task, identity, background, and past interactions
5. **Tools** - Resources the agent can use across various applications

### Task Lifecycle
- Tasks are initialized with defined goals.
- Tasks concurrently monitor triggers:
  - **Notification**: A new message or application update requiring attention
  - **Timeout**: Inactivity prompting follow-up actions
- The trigger system processes notifications and timeouts.
- In each action cycle, you will:
  * Think and plan relevant tool calls
  * Execute these tool calls (e.g., collecting data, posting updates)
  * End the action cycle and await the next trigger
- Tasks conclude when goals are explicitly marked complete

## Current Task

**Goal:** {long_term_goal}

## Memory

{memory}
"""

def get_input_message(last_action_time: str, current_trigger_time: str, trigger_type: str, trigger_content: str, tool_names: str) -> str:
    return f"""You have received a new trigger. A new action cycle has started.

You may perform a series of tool calls to address this trigger.

## Trigger Details

- **Trigger Type:** {trigger_type}
- **Trigger Content:** {trigger_content}

## Available Tools

{tool_names}

## Guidelines

1. Plan and evaluate tool calls with `think()` before and after execution.
2. Complete all necessary actions for the trigger before ending the action cycle.
3. End the action cycle with `end_action_cycle()` or mark the task done with `complete_task()`.
"""

def think(thought: str, log_file: str = None) -> None:
    r"""Use this tool to plan future tool calls or evaluate results.

    This action doesn't change the environment or fetch new data; it simply logs your thought.

    Args:
        thought (str): The thought to document.
    """
    if log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "INFO"
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {level} - Thinking..." + '\n')
            f.write(f"{timestamp} - {level} - {thought}" + '\n')
    else:
        logger.info("Thinking...")
        logger.info(thought)

def end_action_cycle(end_reason: str, log_file: str = None) -> None:
    r"""Ends the current action cycle, specifying why it concluded.

    Args:
        end_reason (str): Explanation for ending the action cycle.
    """
    if log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "INFO"
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {level} - Action Cycle Ended." + '\n')
            f.write(f"{timestamp} - {level} - Reason: {end_reason}" + '\n')
    else:
        logger.info("Action Cycle Ended.")
        logger.info(f"Reason: {end_reason}")

def complete_task(explanation: str, log_file: str = None) -> None:
    r"""Marks the task as completed with a given explanation.

    Args:
        explanation (str): Explanation of task completion.
    """
    if log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = "INFO"
        with open(log_file, "a") as f:
            f.write(f"{timestamp} - {level} - Task Completed." + '\n')
            f.write(f"{timestamp} - {level} - Explanation: {explanation}" + '\n')
    else:
        logger.info("Task Completed.")
        logger.info(f"Explanation: {explanation}")

def get_tool_names(schema: dict) -> str:
    tool_names = []
    for key, value in schema.items():
        if "function" in value:
            func_name = value["function"]["name"]
            tool_names.append(f"{func_name}()")
    return "\n".join(tool_names)