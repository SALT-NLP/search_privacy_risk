from typing import Dict, Optional
import logging
from datetime import datetime
from openai import AzureOpenAI, OpenAIError, RateLimitError
import argparse
from applications import InprocAppClient
import uuid
from utils import retry, unified_call_inproc, parse_response, parse_function_call
from copy import deepcopy
from prompt_utils import (
    get_app_instruction_prompt,
    get_agent_instruction_prompt
)
from agent_utils import (
    think,
    end_action_cycle,
    complete_task,
    get_multi_turn_system_message,
    get_input_message,
    get_tool_names
)
import json
import timeout_decorator

import litellm
from camel.models import LiteLLMModel
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from camel.agents._types import ToolCallRequest
from camel.types.agents import ToolCallingRecord
from camel.toolkits.function_tool import get_openai_tool_schema
from camel.models.model_manager import ModelProcessingError

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def custom_log(message: str, log_file: Optional[str] = None, level: str = "INFO"):
    """Custom logging function that writes to both console and file if specified"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {level} - {message}"

    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
    else:
        print(log_message)

class Task:
    """Class to encapsulate task state and management."""
    def __init__(self, task_id: str, goal: str, max_actions: Optional[int] = 10, time_limit: Optional[int] = 360, response_timeout: Optional[int] = 60):
        self.task_id = task_id
        self.goal = goal
        self.max_actions = int(max_actions)
        self.time_limit = int(time_limit)
        self.actions_taken = 0
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()
        # self.last_update_time = datetime(2025, 1, 1, 0, 0, 0) # initialize to a old time
        self.is_active = True
        self.action_task = None
        self.response_timeout = int(response_timeout)

class Agent(InprocAppClient):
    def __init__(self, agent_id: str, user_id: str, host: str, port: int, deployment_name: str = "azure/gpt-4.1-mini-250414-13576", log_file: Optional[str] = None):
        super().__init__()
        self.agent_id = agent_id
        self.user_id = user_id
        self.host = host
        self.port = port
        self.memory = ""
        self.think_action_history = []
        self.log_file = log_file
        self.llm = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = deployment_name

        self.tasks = {}  # task_id -> Task object

        self.action_type = "multi_turn"

        if self.action_type == "multi_turn":
            if "gemini" in self.deployment_name and "flash" in self.deployment_name:
                model_config_dict = {
                    "reasoning_effort": "disable",
                    "temperature": 1.0,
                    "parallel_tool_calls": False,
                    "drop_params": True
                }
            elif "gpt" in self.deployment_name:
                model_config_dict = {
                    "temperature": 1.0,
                    "parallel_tool_calls": False,
                    "drop_params": True
                }
            else:
                raise NotImplementedError(f"Model {self.deployment_name} not supported")
            
            self.camel_model =  LiteLLMModel(
                model_type=deployment_name,
                model_config_dict=model_config_dict
            )
            custom_log(f"Camel model initialized with deployment name: {deployment_name}", self.log_file)
            custom_log(f"Camel model config: {self.camel_model.model_config_dict}", self.log_file)
            custom_log(f"Camel model type: {self.camel_model.model_type}", self.log_file)
        else:
            raise NotImplementedError(f"Action type {self.action_type} not implemented")
        

        custom_log("###########################################################", self.log_file)
        custom_log(f"Agent {self.agent_id} initialized at {self.host}:{self.port}", self.log_file)
        custom_log(self.get_api_spec(), self.log_file)
        custom_log("###########################################################", self.log_file)
        custom_log("", self.log_file)

    def execute_instruction_on_apps(self, instruction: str) -> Dict:
        return self._process_instruction_on_apps_llm_based(instruction)
        
    def execute_instruction_on_agent(self, instruction: str) -> Dict:
        return self._process_instruction_on_agent_llm_based(instruction)

    def register_app(self, app_name: str, host: str, port: int, auth_token: str) -> Dict:
        self.apps[app_name] = {"host": host, "port": port}
        self.auth_tokens[app_name] = auth_token
        custom_log(f"Registered app {app_name} at {host}:{port}", self.log_file)
        return {"status": "success"}

    def get_health(self) -> Dict:
        return
        
    def init_memory(self, memory: str) -> Dict:
        r"""Initialize the memory of the agent.

        Args:
            memory (str): The memory of the agent
            
        Returns:
            status (str): The status of the operation
            memory (str): The memory of the agent
        """
        self.memory = memory
        custom_log(f"Initialized memory: {memory[:100]}...", self.log_file)
        return {"status": "success", "memory": self.memory}

    def update_memory(self, memory: str) -> Dict:
        r"""Update the memory of the agent.

        Args:
            memory (str): The memory of the agent
        """
        self.memory += f"\n{memory}"
        custom_log(f"Updated memory: {memory[:100]}...", self.log_file)
        return {"status": "success", "memory": self.memory}

    def get_memory(self) -> Dict:
        r"""Get the memory of the agent.

        Returns:
            memory (str): The memory of the agent
        """
        return {"memory": self.memory}

    def start_task(self, goal: str, max_actions: Optional[int] = 10, time_limit: Optional[int] = 360, response_timeout: Optional[int] = 60) -> Dict:
        r"""Start a task with a specific goal. It will monitor the environment for changes (every few seconds) and take proactive actions (depends on the response timeout parameter).
            
        Args:
            goal (str): The goal of the task
            max_actions (int, optional): Maximum number of actions to take. Default is 10.
            time_limit (int, optional): Maximum time to spend on the task in seconds. Default is 360.
            response_timeout (int, optional): Maximum time to wait for taking proactive actions in seconds. Default is 60. If set to 0, proactive actions are disabled.

        Returns:
            status (str): The status of the operation
            message (str): The message of the operation
            task_id (str): The ID of the task
            limits (dict): The limits of the task
        """
        task_id = str(uuid.uuid4())
        task = Task(task_id, goal, max_actions, time_limit, response_timeout)
        self.tasks[task_id] = task
        custom_log(f"Started task {task_id} with goal: {goal}", self.log_file)
            
        return {
            "status": "success", 
            "message": "Task started",
            "task_id": task_id,
            "limits": {
                "max_actions": max_actions,
                "time_limit": time_limit,
                "response_timeout": response_timeout
            }
        }

    def stop_task(self, task_id: str) -> Dict:
        r"""Stop a specific task.

        Args:
            task_id (str): The ID of the task

        Returns:
            status (str): The status of the operation
            message (str): The message of the operation
        """
        if task_id in self.tasks:
            self._stop_task(task_id)
            custom_log(f"Stopped task {task_id}", self.log_file)
            return {"status": "success", "message": f"Task {task_id} stopped"}
        custom_log(f"Failed to stop task {task_id}: Task not found", self.log_file, "ERROR")
        return {"status": "error", "message": f"Task {task_id} not found"}
            
    def list_tasks(self) -> Dict:
        r"""List all active tasks.

        Returns:
            tasks (dict): The tasks
        """
        task_info = {}
        for task_id, task in self.tasks.items():
            if task.is_active:
                elapsed_time = (datetime.now() - task.start_time).total_seconds()
                task_info[task_id] = {
                    "goal": task.goal,
                    "actions_taken": task.actions_taken,
                    "max_actions": task.max_actions,
                    "time_limit": task.time_limit,
                    "elapsed_time": elapsed_time,
                    "is_active": task.is_active
                }
        return {"tasks": task_info}

    def get_task_status(self) -> Dict:
        r"""Get the number of active task and inactive task.

        Returns:
            active_tasks (int): The number of active tasks
            inactive_tasks (int): The number of inactive tasks
        """
        res = {
            "active_tasks": len([task for task in self.tasks.values() if task.is_active]),
            "inactive_tasks": len([task for task in self.tasks.values() if not task.is_active])
        }
        return res
        
    def get_action_info(self) -> Dict:
        r"""Get the action info of the agent.

        Returns:
            actions_taken (int): Number of actions token
        """
        actions_taken = 0
        for task_id, task in self.tasks.items():
            actions_taken += task.actions_taken
        return {"actions_taken": actions_taken}

    @retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, json.JSONDecodeError, AttributeError, TypeError))
    def _process_instruction_on_apps_llm_based(self, instruction: str) -> Dict:
        """Process user instruction using LLM, predict the app and action, and execute corresponding actions."""
        API_spec = ""
        for app_name in self.apps:
            # get the API specification of the app
            app_spec = self.call_app_function(app_name, "get_api_spec")
            app_spec = app_spec["api_spec"]
            API_spec += f"{app_spec}\n"
        
        # Predict the app and action using the prompt
        prompt = get_app_instruction_prompt(API_spec, self.memory, instruction)
        custom_log(f"LLM prompt: {prompt}", self.log_file)

        response = litellm.completion(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1000
        )
        response_content = response.choices[0].message.content.strip()
        custom_log(f"LLM response: {response_content}", self.log_file)

        action_dict = parse_response(response_content)
        function_call = parse_function_call(action_dict)
        if function_call:
            return self.call_app_function(function_call["app"], function_call["function"], **function_call["parameters"])
        else:
            return {"error": "No function call found in the response"}

    @retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, json.JSONDecodeError, AttributeError, TypeError, KeyError))
    def _process_instruction_on_agent_llm_based(self, instruction: str) -> Dict:
        """Process user instruction using LLM, predict the action, and execute corresponding actions."""
        agent_spec = self.get_api_spec()
        prompt = get_agent_instruction_prompt(agent_spec, self.memory, instruction)
        custom_log(f"LLM prompt: {prompt}", self.log_file)

        response = litellm.completion(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1000
        )
        response_content = response.choices[0].message.content.strip()
        custom_log(f"LLM response: {response_content}", self.log_file)

        action_dict = parse_response(response_content)
        function_call = parse_function_call(action_dict)
        if function_call is None:
            # retry
            custom_log("No function call found in the response, retrying...", self.log_file, "ERROR")
            raise KeyError("No function call found in the response")

        if function_call:
            if "agent" not in function_call:
                # retry
                custom_log("Agent not found in the function call, retrying...", self.log_file, "ERROR")
                raise KeyError("Agent not found in the function call")

            if "function" not in function_call:
                # retry
                custom_log("Function not found in the function call, retrying...", self.log_file, "ERROR")
                raise KeyError("Function not found in the function call")

            if "parameters" not in function_call:
                # retry
                custom_log("Parameters not found in the function call, retrying...", self.log_file, "ERROR")
                raise KeyError("Parameters not found in the function call")

            if function_call["function"] == "init_memory":
                if "memory" not in function_call["parameters"]:
                    # retry
                    custom_log("Memory not found in the parameters, retrying...", self.log_file, "ERROR")
                    raise KeyError("Memory not found in the parameters")

                return self.init_memory(**function_call["parameters"])
            elif function_call["function"] == "update_memory":
                return self.update_memory(**function_call["parameters"])
            elif function_call["function"] == "get_memory":
                return self.get_memory()
            elif function_call["function"] == "start_task":
                if "goal" not in function_call["parameters"]:
                    # retry
                    custom_log("Goal not found in the parameters, retrying...", self.log_file, "ERROR")
                    raise KeyError("Goal not found in the parameters")

                return self.start_task(**function_call["parameters"])
            elif function_call["function"] == "stop_task":
                return self.stop_task(**function_call["parameters"])
            elif function_call["function"] == "list_tasks":
                return self.list_tasks()
            else:
                custom_log(f"Function call not implemented: {function_call['function']}", self.log_file, "ERROR")
                raise KeyError(f"Function call not implemented: {function_call['function']}, retrying...")

    def _check_task_conditions(self, task_id: str) -> bool:
        """Check all task stopping conditions."""
        if task_id not in self.tasks:
            return True
            
        task = self.tasks[task_id]
        
        # Check time limit
        if task.time_limit:
            elapsed_time = (datetime.now() - task.start_time).total_seconds()
            if elapsed_time >= task.time_limit:
                custom_log(f"Time limit of {task.time_limit} seconds reached. Stopping task {task_id}.", self.log_file)
                self._stop_task(task_id)
                return True

        # Check action limit
        if task.max_actions and task.actions_taken >= task.max_actions:
            custom_log(f"Maximum number of actions ({task.max_actions}) reached. Stopping task {task_id}.", self.log_file)
            self._stop_task(task_id)
            return True

        return False

    def _stop_task(self, task_id: str):
        """Internal method to stop a specific task."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        
        if task.action_task:
            task.action_task.cancel()
            task.action_task = None
        
        task.is_active = False

    def start_action_cycle(self, task_id: str):
        """Monitor for any responses or changes in the environment every few seconds."""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]

        if not task.is_active:
            custom_log(f"Task {task_id} is not active, skipping action cycle", self.log_file)
            return
        
        is_there_new_activity = False
        new_activity_descriptions_list = []

        # Create a list of tasks
        results = [
            self.call_app_function(app_name, "get_new_activity", since=task.last_update_time)
            for app_name in self.apps
        ]

        current_time = datetime.now()

        for app_name, result in zip(self.apps, results):
            has_new_activity = result.get("has_new_activity", False)

            if has_new_activity:
                is_there_new_activity = True
                new_activity_descriptions = result.get("new_activity_descriptions", [])
                if app_name == "Gmail":
                    new_activity_descriptions_list.append(f"{len(new_activity_descriptions)} new emails on Gmail")
                elif app_name == "Facebook":
                    new_activity_descriptions_list.append(f"{len(new_activity_descriptions)} new posts on Facebook")
                elif app_name == "Messenger":
                    new_activity_descriptions_list.append(f"{len(new_activity_descriptions)} new messages on Messenger")
                elif app_name == "Notion":
                    new_activity_descriptions_list.append(f"{len(new_activity_descriptions)} new shared pages on Notion")
                else:
                    print(f"Unknown app: {app_name}")

        if is_there_new_activity:
            new_activity_descriptions_list_str = ", ".join(new_activity_descriptions_list)

            custom_log(f"New activity detected, {new_activity_descriptions_list_str}, for task {task_id}, taking new action", self.log_file)

            trigger_type = "notification"
            trigger_content = new_activity_descriptions_list_str
        else:
            if task.response_timeout > 0 and task.actions_taken > 0:
                custom_log(f"Response timeout reached for task {task_id}, taking new action", self.log_file)

                trigger_type = "timeout"
                trigger_content = f"There has been no notification for a while, take more proactive actions or mark the task as completed."
            elif task.response_timeout > 0 and task.actions_taken == 0:
                custom_log("The first action cycle of the task is starting...", self.log_file)

                trigger_type = "timeout"
                trigger_content = f"This is the first action cycle of the task. Take proactive actions to achieve the goal."
            else:
                custom_log(f"Task {task_id} has no response timeout, skipping action cycle, last update time: {task.last_update_time}, current time: {current_time}", self.log_file)
                return

        last_action_time = str(task.last_update_time)
        task.last_update_time = current_time

        if self.action_type == "multi_turn":
            action_result = self._take_multi_turn_action(task_id, str(last_action_time), str(current_time), trigger_type, trigger_content)
        else:
            raise NotImplementedError(f"Action type {self.action_type} not implemented")

        if action_result == "COMPLETE":
            custom_log(f"Task {task_id} completed successfully.", self.log_file)
            self._stop_task(task_id)
            return

    def _take_multi_turn_action(self, task_id: str, last_action_time: str, current_trigger_time: str, trigger_type: Optional[str] = None, trigger_content: Optional[str] = None) -> Optional[str]:
        """Generate and take actions based on the task goal by multi-turn conversation"""
        def tool_call_to_function_call(tool_call_request: ToolCallRequest) -> dict:
            custom_log(f"Tool call to function call: {tool_call_request}", self.log_file)
            app_name = tool_call_request.tool_name.split("_")[0]
            function_name = "_".join(tool_call_request.tool_name.split("_")[1:])
            custom_log(f"App name: {app_name}, Function name: {function_name}", self.log_file)
            custom_log(f"Tool call arguments: {tool_call_request.args}", self.log_file)
            return {
                "app": app_name,
                "function": function_name,
                "parameters": tool_call_request.args
            }

        def execute_tool(agent: ChatAgent, tool_call_request: ToolCallRequest) -> ToolCallingRecord:
            r"""Execute the tool with arguments following the model's response.

            Args:
                tool_call_request (ToolCallRequest): The tool call request.

            Returns:
                FunctionCallingRecord: A struct for logging information about this
                    function call.
            """
            func_name = tool_call_request.tool_name
            args = tool_call_request.args
            tool_call_id = tool_call_request.tool_call_id

            try:
                if func_name not in ["think", "end_action_cycle", "complete_task"]:
                    function_call = tool_call_to_function_call(tool_call_request)
                    result = self.call_app_function(
                        function_call["app"], 
                        function_call["function"], 
                        **function_call["parameters"]
                    )

                    # remove receiver_id, activity_id, activity_description from the result
                    # This is only for activity logs
                    # For consistency with the API spec
                    if 'receiver_id' in result and 'activity_id' in result and 'activity_description' in result:
                        del result['receiver_id']
                        del result['activity_id']
                        del result['activity_description']

                else:
                    temp_args = deepcopy(args)
                    temp_args["log_file"] = self.log_file

                    if func_name == "think":
                        result = think(**temp_args)
                    elif func_name == "end_action_cycle":
                        result = end_action_cycle(**temp_args)
                    elif func_name == "complete_task":
                        result = complete_task(**temp_args)
            except Exception as e:
                # Capture the error message to prevent framework crash
                error_msg = f"Error executing tool '{func_name}': {e!s}"
                result = {"error": error_msg}

            custom_log(f"Result: {result}", self.log_file)
            return agent._record_tool_calling(func_name, args, result, tool_call_id)
        
        # Timeout decorator might not work properly
        @timeout_decorator.timeout(120, timeout_exception=TimeoutError)
        def agent_step_with_tool_call(agent: ChatAgent, input_message: str) -> None:
            think_step = True
            task_completed = False

            initial_time = datetime.now()
            agent_step_count = 0

            while True:
                if think_step:
                    # At the thinking step, force the model to think
                    agent.model_backend.current_model.model_config_dict["tool_choice"] = {"type": "function", "function": {"name": "think"}}
                    response = agent.step(input_message)
                    del agent.model_backend.current_model.model_config_dict["tool_choice"]
                else:
                    # At the action step, force the model to call the tool
                    agent.model_backend.current_model.model_config_dict["tool_choice"] = "required"
                    response = agent.step(input_message)
                    del agent.model_backend.current_model.model_config_dict["tool_choice"]

                custom_log(f"Response: {response}", self.log_file)

                cycle_finished = False
                for tool_call_request in response.info["external_tool_call_requests"]:
                    execute_tool(agent, tool_call_request)
                    if tool_call_request.tool_name == "end_action_cycle":
                        cycle_finished = True
                    elif tool_call_request.tool_name == "complete_task":
                        cycle_finished = True
                        task_completed = True

                # Toggle the think step
                think_step = not think_step
                agent_step_count += 1

                if (datetime.now() - initial_time).total_seconds() > 120 or agent_step_count > 50:
                    custom_log(f"Timeout in action cycle, time taken: {(datetime.now() - initial_time).total_seconds()}, agent step count: {agent_step_count}", self.log_file)
                    print(f"Timeout in action cycle, time taken: {(datetime.now() - initial_time).total_seconds()}, agent step count: {agent_step_count}")
                    break

                if cycle_finished:
                    break
                input_message = agent.memory.get_context()[0][-1]
            
            return task_completed

        def agent_step_with_tool_call_with_retry(agent: ChatAgent, input_message: str) -> None:
            try:
                return agent_step_with_tool_call(agent, input_message)
            except (TimeoutError, json.JSONDecodeError, ModelProcessingError, TypeError):
                return False

        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        
        app_spec_dict = {}

        extra_tools = [think, end_action_cycle, complete_task]
        for tool in extra_tools:
            tool_schema = get_openai_tool_schema(tool)

            # remove log_file from tool_schema
            if 'log_file' in tool_schema['function']['parameters']['properties']:
                del tool_schema['function']['parameters']['properties']['log_file']
            tool_schema['function']['parameters']['required'] = [param for param in tool_schema['function']['parameters']['required'] if param != 'log_file']

            app_spec_dict[tool.__name__] = tool_schema

        # Get API specs for all apps
        for app_name in self.apps:
            app_spec = self.call_app_function(app_name, "get_api_spec_in_openai_tool_schema")
            app_spec = app_spec["api_spec"]
            for key, value in app_spec.items():
                app_spec_dict[key] = value
                
        system_message = get_multi_turn_system_message(self.memory, task.goal, task.start_time, task.time_limit)
        agent = ChatAgent(model=self.camel_model, system_message=system_message)

        # Load trajectory from previous action cycles
        for privious_action_cycle in self.think_action_history:
            agent.load_memory(privious_action_cycle, skip_system_message=True)

        agent._external_tool_schemas = app_spec_dict

        custom_log("Tools:", self.log_file)
        custom_log(agent._external_tool_schemas, self.log_file)

        tool_names = get_tool_names(agent._external_tool_schemas)
        custom_log(f"Tool names: {tool_names}", self.log_file)

        input_message = get_input_message(last_action_time, current_trigger_time, trigger_type, trigger_content, tool_names)
        task_completed = agent_step_with_tool_call_with_retry(agent, input_message)

        custom_log("Agent Memory:", self.log_file)
        custom_log(agent.memory.get_context(), self.log_file)

        # Save trajectory for future action cycles
        self.think_action_history.append(deepcopy(agent.memory))

        if not task_completed:
            # Cycle is complete but task goal is not completed
            final_response = "CYCLE_COMPLETE"
        else:
            # Task goal is completed
            final_response = "COMPLETE"

        task.actions_taken += 1
        custom_log(f"Task {task_id}: {task.actions_taken} actions taken", self.log_file)

        if task.actions_taken >= task.max_actions:
            custom_log(f"Maximum number of actions ({task.max_actions}) reached. Stopping task {task_id}.", self.log_file)
            final_response = "COMPLETE"

        return final_response

    def get_api_spec(self) -> str:
        """Returns a string describing the API specification of this Agent instance"""

        routes_spec = []
        for function in [self.init_memory, self.update_memory, self.get_memory, self.start_task, self.stop_task, self.list_tasks]:
            docstring = function.__doc__ or ""
            docstring = "\n".join([item.strip() for item in docstring.split("\n")])
            routes_spec.append(f"{len(routes_spec) + 1}. {function.__name__}\n"
                                f"Description: {docstring.strip()}\n")

        spec = f"""
Agent: {self.agent_id}

Available Endpoints:
{chr(10).join(routes_spec)}
"""
        return spec.strip()