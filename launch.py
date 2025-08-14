import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
from fastapi.testclient import TestClient
import utils
from applications import Facebook, Notion, Gmail, Messenger
from agent import Agent
from user import user_init
import sys
from utils import get_info_inproc

def read_commands_from_file(file_path: str) -> List[Tuple[str, str]]:
    """Load [(command, log_name), ...] from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)
    
def agent_step(agent, task_id, stdout=None):
    print(f"{agent.agent_id}: Starting action cycle for task {task_id}...", file=stdout)
    agent.start_action_cycle(task_id)

def user_step(user, agent_dict, stdout=None):
    if user.current_pointer < len(user.commands):
        command = user.commands[user.current_pointer].strip()
        if len(command) == 0:
            user.current_pointer += 1
            return {"status": "success", "message": "Empty command"}
        elif command.startswith("wait"):
            port = command.split("localhost:")[1].strip().split("/")[0]
            function = command.split(f"http://localhost:{port}/")[1].strip().split(" ")[0]
            arguments = command.split(f"http://localhost:{port}/{function} ")[1].strip()
            if function == "get_task_status":
                if "active_tasks=0 inactive_tasks=1" in arguments:
                    for agent_id in agent_dict:
                        if int(agent_dict[agent_id].port) == int(port):
                            task_status = agent_dict[agent_id].get_task_status()
                            if task_status["active_tasks"] == 0 and task_status["inactive_tasks"] == 1:
                                print(f"{user.user_id}: The agent {agent_id} has finished its task...", file=stdout)
                                user.current_pointer += 1
                                return {"status": "success", "message": "The agent has finished its task"}
                            else:
                                print(f"{user.user_id}: The agent {agent_id} has not finished its task...", file=stdout)
                                return {"status": "error", "message": f"The agent {agent_id} has not finished its task..."}
                else:
                    raise NotImplementedError(f"Command not implemented: {command}")
            else:
                raise NotImplementedError(f"Command not implemented: {command}")
        elif command.startswith("sleep"):
            user.current_pointer += 1
            print(f"{user.user_id}: Sleeping skipped. Continuing to the next command...", file=stdout)
            # continue to the next command
            return user_step(user, agent_dict)
        elif command.startswith("end simulation"):
            user.current_pointer += 1
            return {"status": "success", "message": "Skipped for inproc..."}
        elif "[AGENT]" in command:
            agent_id = command.split("[")[1].split("]")[0]
            command = command.split("[AGENT]")[1].strip()
            result = agent_dict[agent_id].execute_instruction_on_agent(command)
            user.current_pointer += 1
            print(f"{user.user_id}: {agent_id} executed: {command}... {result}", file=stdout)
            return result
        else:
            raise NotImplementedError(f"Command not implemented: {command}")
    else:
        print(f"{user.user_id}: No more commands to execute...", file=stdout)
        return None

def run_experiment_appless(commands_file: str, exp_id: str, duration: int = 240, minimal_action_taken: int = 5, stdout=None):
    """
    Launch all commands in *commands_file* in-process, log outputs to log_<exp_id>/,
    keep them alive for *duration* seconds, then clean up.
    """
    commands_path = Path(commands_file).expanduser().resolve()
    base_dir = commands_path.parent
    log_dir = base_dir / f"log_{exp_id}"
    log_dir.mkdir(exist_ok=True)

    commands = read_commands_from_file(str(commands_path))
    threads = []
    user_threads = []

    sender_agent_id = None
    recipient_agent_id_list = []

    agent_dict = {}
    user_dict = {}
    for cmd, log in commands:
        log_path = log_dir / Path(log).name
        cmd = cmd.replace("--db-folder logs", f"--db-folder {log_dir}")

        if "application.py" in cmd:
            # Extract port from command
            port = int(cmd.split("--port")[1].strip().split(" ")[0])
            app_name = cmd.split("--app-name")[1].strip().split(" ")[0]
            db_folder = cmd.split("--db-folder")[1].strip().split(" ")[0]
            
            # Create the appropriate application based on the app type
            if app_name == "Facebook":
                app = Facebook(app_name, "localhost", port, db_folder)
            elif app_name == "Notion":
                app = Notion(app_name, "localhost", port, db_folder)
            elif app_name == "Gmail":
                app = Gmail(app_name, "localhost", port, db_folder)
            elif app_name == "Messenger":
                app = Messenger(app_name, "localhost", port, db_folder)
            else:
                raise ValueError(f"Unknown application name: {app_name}")
            
            utils.clients_inproc[port] = TestClient(app.app)
            
        elif "agent.py" in cmd:
            port = int(cmd.split("--port")[1].strip().split(" ")[0])
            agent_id = cmd.split("--agent-id")[1].strip().split(" ")[0].strip()
            user_id = cmd.split("--user-id")[1].strip().split(" ")[0].strip()
            deployment_name = cmd.split("--deployment-name")[1].strip().split(" ")[0].strip()
            agent_dict[agent_id] = Agent(agent_id, user_id, "localhost", port, deployment_name, log_path)
            
        elif "data_sender" in cmd:
            sender_agent_id = cmd.split("--agent-id")[1].strip().split(" ")[0].strip()
        elif "data_recipient" in cmd:
            recipient_agent_id_list.append(cmd.split("--agent-id")[1].strip().split(" ")[0].strip())

    for cmd, log in commands:
        log_path = log_dir / Path(log).name
        
        cmd = re.sub(
            r"--init-file\s+(\w+)\.txt",
            lambda m: f'--init-file {base_dir / (m.group(1) + ".txt")}',
            cmd,
        )
        cmd = re.sub(
            r"--access-control-file\s+(\w+)\.json",
            lambda m: f'--access-control-file {base_dir / (m.group(1) + ".json")}',
            cmd,
        )

        if "user.py" in cmd:
            user_id = cmd.split("--user-id")[1].strip().split("--")[0].strip()
            username = cmd.split("--username")[1].strip().split("--")[0].strip()
            app_hosts = [host.strip() for host in cmd.split("--app-host")[1].strip().split("--")[0].split(" ") if host.strip()]
            app_ports = [port.strip() for port in cmd.split("--app-port")[1].strip().split("--")[0].split(" ") if port.strip()]
            app_names = [name.strip() for name in cmd.split("--app-name")[1].strip().split("--")[0].split(" ") if name.strip()]
            agent_hosts = [host.strip() for host in cmd.split("--agent-host")[1].strip().split("--")[0].split(" ") if host.strip()]
            agent_ports = [port.strip() for port in cmd.split("--agent-port")[1].strip().split("--")[0].split(" ") if port.strip()]
            agent_ids = [id.strip() for id in cmd.split("--agent-id")[1].strip().split("--")[0].split(" ") if id.strip()]
            init_file = cmd.split("--init-file")[1].strip()
            print(app_hosts, app_ports, app_names, agent_hosts, agent_ports, agent_ids, init_file, file=stdout)
            user = user_init(user_id, username, app_hosts, app_ports, app_names, agent_hosts, agent_ports, agent_ids, agent_dict, init_file, log_file=log_path)
            user_dict[user_id] = user

    sender_action_taken_function = agent_dict[sender_agent_id].get_action_info
    sender_status_function = agent_dict[sender_agent_id].get_task_status

    recipient_action_taken_function_list = [agent_dict[agent_id].get_action_info for agent_id in recipient_agent_id_list]
    recipient_status_function_list = [agent_dict[agent_id].get_task_status for agent_id in recipient_agent_id_list]

    def get_overall_status():
        sender_status = sender_status_function()
        recipient_status_list = [status_function() for status_function in recipient_status_function_list]
        
        sender_action_info = sender_action_taken_function()
        recipient_action_info_list = [action_taken_function() for action_taken_function in recipient_action_taken_function_list]
        
        if "active_tasks" in sender_status and "inactive_tasks" in sender_status:
            if sender_status["active_tasks"] == 0 and sender_status["inactive_tasks"] == 1:
                print("The sender agent has no active task...", file=stdout)
                return True, None

        all_recipient_finished = True
        for recipient_status in recipient_status_list:
            if "active_tasks" in recipient_status and "inactive_tasks" in recipient_status:
                if recipient_status["active_tasks"] == 0 and recipient_status["inactive_tasks"] == 1:
                    pass
                else:
                    all_recipient_finished = False
            else:
                all_recipient_finished = False

        if all_recipient_finished:
            print("All recipient agents have finished their tasks...", file=stdout)
            return True, None
        
        sender_action_taken = 0
        recipient_action_taken_list = []
        if "actions_taken" in sender_action_info:
            sender_action_taken = sender_action_info["actions_taken"]
        
        for idx, recipient_action_info in enumerate(recipient_action_info_list):
            if "actions_taken" in recipient_action_info:
                recipient_action_taken_list.append(recipient_action_info["actions_taken"])

        return False, [sender_action_taken, recipient_action_taken_list]

    try:
        initial_time = time.time()
        old_action_info = [0, [0] * len(recipient_agent_id_list)]
        maximum_step = 20
        step = 0
        while True:
            step += 1
            if step > maximum_step:
                print(f"Maximum step reached. Ending simulation... action taken: {action_info}", file=stdout)
                break

            for user in user_dict.values():
                user_step(user, agent_dict, stdout=stdout)
            
            for agent in agent_dict.values():
                active_task_ids = agent.list_tasks()["tasks"].keys()
                for task_id in active_task_ids:
                    agent_step(agent, task_id, stdout=stdout)

            finished, action_info = get_overall_status()
            if finished:
                print(f"Time taken: {time.time() - initial_time} seconds... action taken: {action_info}", file=stdout)
                break
            
            if action_info != old_action_info:
                old_action_info = action_info
                print(f"Time taken: {time.time() - initial_time} seconds... action taken: {action_info}", file=stdout)

            if time.time() - initial_time > duration:
                recipient_action_taken_list = action_info[1]
                if any(item >= minimal_action_taken for item in recipient_action_taken_list):
                    print(f"Duration reached. Enough actions taken. Ending simulation... action taken: {action_info}", file=stdout)
                    break
                else:
                    print(f"Not enough actions taken. Continuing... current action taken: {action_info}", file=stdout)

                if time.time() - initial_time > duration * 2:
                    print(f"Double duration reached. Ending simulation... action taken: {action_info}", file=stdout)
                    break

        print(f"Time taken: {time.time() - initial_time} seconds", file=stdout)
        print(f"Current time: {datetime.now()}", file=stdout)
    except KeyboardInterrupt:
        print("Keyboard interrupt received—cleaning up.", file=stdout)
    finally:
        # Clean up threads
        for thread in user_threads:
            thread.join(timeout=1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch multiple components with commands from a file (in-process version)"
    )
    parser.add_argument("--commands-file", required=True, help="JSON file of commands")
    parser.add_argument("--exp-id", required=True, help="Experiment ID")
    parser.add_argument(
        "--duration", type=int, default=240, help="Run time in seconds (default: 240)"
    )
    parser.add_argument(
        "--minimal-action-taken", type=int, default=5, help="Minimal action taken (default: 5)"
    )
    args = parser.parse_args()

    run_experiment_appless(args.commands_file, args.exp_id, args.duration, args.minimal_action_taken, stdout=None) 