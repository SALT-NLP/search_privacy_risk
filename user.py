from typing import Dict, Optional, List, Set
import re
import logging
import json
from applications import InprocAppClient
from utils import unified_call_inproc
import argparse
import httpx
from utils import wait_for_condition_inproc
from datetime import datetime
from agent import Agent

def custom_log(message: str, log_file: Optional[str] = None, level: str = "INFO"):
    """Custom logging function that writes to both console and file if specified"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {level} - {message}"
    print(log_message)  # Always print to console
    
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")

class UserClient(InprocAppClient):
    def __init__(self, user_id: str, username: str, init_file: Optional[str] = None, access_control_file: Optional[str] = None, \
                 log_file: Optional[str] = None):
        super().__init__()
        self.user_id = user_id
        self.username = username
        self.agents: Dict[str, Dict] = {}  # Store agent connection info
        self.agent_app_access: Dict[str, Set[str]] = {}  # Store which agent has access to which apps
        self.log_file = log_file
        custom_log(f"UserClient {self.user_id} initialized", self.log_file)
        self.init_task = None
        
        # Store init file for later processing
        self.init_file = init_file

        self.current_pointer = 0
        self.commands = []
        
        # Process access control file if provided
        self.access_control_file = access_control_file
        if access_control_file:
            self._load_access_control(access_control_file)
        
    def _load_access_control(self, access_control_file: str):
        """Load access control configuration from file"""
        try:
            with open(access_control_file, 'r') as f:
                access_config = json.load(f)
                
            for agent_id, app_name in access_config.items():
                if isinstance(app_name, list):
                    self.agent_app_access[agent_id] = set(app_name)
                else:
                    self.agent_app_access[agent_id] = {app_name}
                    
            custom_log(f"Loaded access control configuration: {self.agent_app_access}", self.log_file)
        except Exception as e:
            custom_log(f"Error loading access control file: {e}", self.log_file, "ERROR")
            # Default: empty access control config
            self.agent_app_access = {}

    def process_init_file(self) -> Optional[str]:
        """Process initialization file if provided"""
        if self.init_file:
            custom_log(f"\nProcessing initialization file: {self.init_file}", self.log_file)
            result = self._process_init_file(self.init_file)
            return result

    def _process_command(self, command: str) -> Optional[Dict]:
        """Process a single command and return the result if any."""
        try:
            if command.startswith('end simulation'):
                return {"status": "success", "message": "Skipped for inproc..."}
            elif command.startswith('sleep'):
                return {"status": "success", "message": "Skipped for inproc..."}
            elif command.startswith('wait'):
                return {"status": "success", "message": "Skipped for inproc..."}
            # Parse and execute the command
            elif command.startswith('[') and ']' in command:
                # Extract agent_id and the actual command
                agent_part = command[:command.find(']')+1]
                target_agent_id = agent_part[1:-1].strip()
                remaining_command = command[command.find(']')+1:].strip()

                if target_agent_id not in self.agents:
                    custom_log(f"Error: Agent {target_agent_id} not registered", self.log_file, "ERROR")
                    return None

                # Check if this is an APP or AGENT instruction
                if remaining_command.startswith('[APP]'):
                    actual_command = remaining_command[len('[APP]'):].strip()
                    custom_log(f"Sending APP command to agent {target_agent_id}...", self.log_file)
                    return self.send_apps_instruction_to_agent(target_agent_id, actual_command)
                elif remaining_command.startswith('[AGENT]'):
                    actual_command = remaining_command[len('[AGENT]'):].strip()
                    custom_log(f"Sending AGENT command to agent {target_agent_id}...", self.log_file)
                    return self.send_agent_instruction_to_agent(target_agent_id, actual_command)
                else:
                    custom_log("Error: Agent instruction must specify [APP] or [AGENT]", self.log_file, "ERROR")
                    return None
            else:
                # This is a direct app action
                if command.startswith('{') and '}' in command:
                    app_part = command[:command.find('}')+1]
                    app_name = app_part[1:-1].strip()
                    command_to_parse = command[command.find('}')+1:].strip()

                    # Check if app name is specified
                    app_name = self._get_app_name(app_name)
                    if not app_name:
                        custom_log(f"Error: App not registered", self.log_file, "ERROR")
                        return None

                    # Split into action and parameters
                    parts = command_to_parse.split(maxsplit=1)
                    if not parts:
                        custom_log("Error: No action specified", self.log_file, "ERROR")
                        return None
                        
                    action = parts[0]
                    params_str = parts[1] if len(parts) > 1 else ""

                    # To handle quoted strings properly
                    params = {}
                    if params_str:
                        # Match pattern: key="value with spaces" or key=value
                        pattern = r'(\w+)=(?:"([^"]*)"|\b(\S+)\b)'
                        matches = re.finditer(pattern, params_str)
                        
                        for match in matches:
                            key = match.group(1)
                            # group(2) is the quoted value, group(3) is the unquoted value
                            value = match.group(2) if match.group(2) is not None else match.group(3)
                            params[key] = value

                    custom_log(f"Executing direct app action on {app_name}: {action}", self.log_file)
                    return self.direct_app_action(app_name, action, **params)
                else:
                    custom_log("Error: Invalid command format", self.log_file, "ERROR")
                    return None
                    
        except httpx.ReadTimeout:
            custom_log("Error: Request timed out. The server took too long to respond.", self.log_file, "ERROR")
            return {"status": "error", "message": "Request timed out"}
        except Exception as e:
            custom_log(f"Error in _process_command: {e}", self.log_file, "ERROR")
            custom_log(f"Error details - Type: {type(e)}, Args: {e.args}", self.log_file, "ERROR")
            return None
            
    def _process_init_file(self, init_file: str) -> Optional[str]:
        """Process commands from initialization file"""
        try:
            with open(init_file, 'r') as f:
                commands = f.readlines()
            
            for command in commands:
                command = command.strip()
                if not command or command.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                custom_log(f"\nExecuting init command: {command}", self.log_file)
                result = self._process_command(command)
                custom_log(f"Result: {result}", self.log_file)
                custom_log(f"Time: {datetime.now()}", self.log_file)

                if result and result.get("status") == "success" and result.get("message") == "Ending simulation":
                    return "Ending simulation"
            
            return None
                    
        except Exception as e:
            custom_log(f"Error processing initialization file: {e}", self.log_file, "ERROR")

    def register_app(self, app_name: str, host: str, port: int, agent_ids: Optional[List[str]] = None, agent_dict: Dict[str, Agent] = None) -> Dict:
        """Register an application with this client and specified agents
        
        Args:
            app_name: Name of the application
            host: Host of the application
            port: Port of the application
            agent_ids: List of agent IDs that should have access to this app.
                       If None, all agents will have access.
        """
        # First register the app info without auth token
        self.register_app_info(app_name, host, port)
        
        # Authenticate with the app
        result = self._try_app_auth(app_name, self.user_id, self.username, "default_password")
        
        # Store the auth token
        self.auth_tokens[app_name] = result["auth_token"]
        
        # Determine which agents should have access to this app
        target_agents = {}
        if agent_ids is None:
            # By default, all agents have access
            target_agents = self.agents
        else:
            # Only specified agents have access
            for agent_id in agent_ids:
                if agent_id in self.agents:
                    target_agents[agent_id] = self.agents[agent_id]
                    
                    # Update access control mapping
                    if agent_id not in self.agent_app_access:
                        self.agent_app_access[agent_id] = set()
                    self.agent_app_access[agent_id].add(app_name)
                else:
                    custom_log(f"Agent {agent_id} not registered, cannot grant access to {app_name}", self.log_file, "WARNING")
        
        # Register app with target agents
        custom_log(f"Registering app {app_name} at {host}:{port} with {len(target_agents)} agents", self.log_file)
        for agent_id, agent_info in target_agents.items():
            try:
                # Check if this agent should have access based on access control file
                if self.access_control_file and agent_id in self.agent_app_access:
                    if app_name not in self.agent_app_access[agent_id]:
                        custom_log(f"Skipping registration of app {app_name} with agent {agent_id} due to access control", self.log_file)
                        continue
                
                agent_dict[agent_id].register_app(app_name, host, port, result["auth_token"])
                custom_log(f"Successfully registered app {app_name} with agent {agent_id}", self.log_file)
            except Exception as e:
                custom_log(f"Failed to register app {app_name} with agent {agent_id}: {e}", self.log_file, "ERROR")
                
        return {"status": "success", "auth_token": result["auth_token"]}

    def register_agent(self, agent_id: str, host: str, port: int) -> Dict:
        """Register an existing agent's connection information"""
        self.agents[agent_id] = {"host": host, "port": port}
        
        # Initialize empty app access set if not already present
        if agent_id not in self.agent_app_access:
            self.agent_app_access[agent_id] = set()
            
        return {"agent_id": agent_id, "host": host, "port": port}
        
    def send_apps_instruction_to_agent(self, agent_id: str, instruction: str) -> Dict:
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
            
        agent_info = self.agents[agent_id]
        base_url = f"http://{agent_info['host']}:{agent_info['port']}"
        return unified_call_inproc(base_url, "execute_instruction_on_apps", instruction=instruction)
    
    def send_agent_instruction_to_agent(self, agent_id: str, instruction: str) -> Dict:
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
            
        agent_info = self.agents[agent_id]
        base_url = f"http://{agent_info['host']}:{agent_info['port']}"
        return unified_call_inproc(base_url, "execute_instruction_on_agent", instruction=instruction)

    def direct_app_action(self, app_name: str, action: str, **kwargs) -> Dict:
        return self.call_app_function(app_name, action, **kwargs)

def user_init(
    user_id: str,
    username: str,
    app_hosts: List[str],
    app_ports: List[int],
    app_names: List[str],
    agent_hosts: List[str],
    agent_ports: List[int],
    agent_ids: List[str],
    agent_dict: Dict[str, Agent],
    init_file: Optional[str] = None,
    access_control_file: Optional[str] = None,
    log_file: Optional[str] = None
) -> Optional[str]:
    # Create UserClient
    user = UserClient(user_id, username, init_file, access_control_file, log_file)
    
    # Register all agents
    for i, agent_id in enumerate(agent_ids):
        if not agent_id or i >= len(agent_hosts) or i >= len(agent_ports) or not agent_hosts[i] or not agent_ports[i]:
            custom_log(f"Warning: Agent {i+1} (ID: {agent_id}) has invalid host or port", None, "WARNING")
            continue

        custom_log(f"\nRegistering agent {agent_id}...", None)
        user.register_agent(agent_id, agent_hosts[i], agent_ports[i])

    # Register all apps
    for i, app_name in enumerate(app_names):
        if not app_name or i >= len(app_hosts) or i >= len(app_ports) or not app_hosts[i] or not app_ports[i]:
            custom_log(f"Warning: App {i+1} (Name: {app_name}) has invalid host or port", None, "WARNING")
            continue
        
        # Determine which agents should have access to this app
        target_agents = None  # None means all agents by default

        custom_log(f"\nRegistering app {app_name}...", None)
        user.register_app(app_name, app_hosts[i], app_ports[i], target_agents, agent_dict)
    
    user.commands = []
    if init_file:
        with open(init_file, 'r') as f:
            user.commands = f.readlines()

    return user