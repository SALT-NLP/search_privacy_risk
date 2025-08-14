import json
import os
from datetime import datetime
from launch import run_experiment_appless
from copy import deepcopy
import asyncio
from utils import get_user_id, get_user_email, find_free_ports
import multiprocessing
from typing import List, Tuple
import contextlib
from tqdm import tqdm

def get_third_person_intro_and_contact(agent_dict, apps):
    intro = agent_dict["public_profile"]
    contact = ""
    for app in apps:
        if app == "Facebook":
            contact += "Facebook ID: " + get_user_id(agent_dict["concrete_name"]) + ", "
        elif app == "Messenger":
            contact += "Messenger ID: " + get_user_id(agent_dict["concrete_name"]) + ", "
        elif app == "Notion":
            contact += "Notion ID: " + get_user_id(agent_dict["concrete_name"]) + ", "
        elif app == "Gmail":
            contact += "Email: " + get_user_email(agent_dict["concrete_name"]) + ", "
    return intro + " (Contact Information: " + contact.strip().rstrip(",") + ")"

def get_user_command(user_id, user_name, social_background, contact, task, init_file_name, apps, agent_role, sensitive_data, app_info_dict, agent_info_dict):
    app_names = " ".join([app for app in apps])
    app_hosts = " ".join(["localhost"] * len(apps))
    app_ports_str = " ".join([str(app_info_dict[app]) for app in apps])

    agent_id = agent_info_dict[agent_role]["id"]
    agent_host = "localhost"
    agent_port_str = str(agent_info_dict[agent_role]["port"])

    with open(os.path.join(current_folder, init_file_name), "w") as f:
        if user_name not in social_background:
            intro = f"You are {user_name}. {social_background}"
        else:
            intro = social_background

        memory = f"{intro} {contact} {sensitive_data}".strip()
        f.write(f"[{agent_id}] [AGENT] init the memory with \"{memory}\"\n")

        if "data_subject" in init_file_name:
            f.write(f"[{agent_id}] [AGENT] start a task with the following goals: \"{task}\". Set the maximum number of actions to 1.\n")
            f.write(f"wait http://localhost:{agent_info_dict['data_subject']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=60\n")
            f.write("end simulation\n")
        elif "data_sender" in init_file_name:
            if args.sensitive_data_in_memory:
                f.write(f"[{agent_id}] [AGENT] start a task with the following goals: \"{task}\". Set the response timeout to 0 to disable proactive actions, and set a time limit of 3600 seconds (1 hour).\n")
                f.write(f"wait http://localhost:{agent_info_dict['data_subject']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=60\n")
            else:
                f.write(f"wait http://localhost:{agent_info_dict['data_subject']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=60\n")
                f.write(f"[{agent_id}] [AGENT] start a task with the following goals: \"{task}\". Set the response timeout to 0 to disable proactive actions, and set a time limit of 3600 seconds (1 hour).\n")
            f.write(f"wait http://localhost:{agent_info_dict['data_recipient']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=180\n")
            f.write("end simulation\n")
        elif "data_recipient" in init_file_name:
            f.write(f"wait http://localhost:{agent_info_dict['data_subject']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=60\n")
            f.write("sleep 5\n")
            f.write(f"[{agent_id}] [AGENT] start a task with the following goals: \"{task}\".\n")
            f.write("sleep 5\n")
            f.write(f"wait http://localhost:{agent_info_dict['data_recipient']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=180\n")
            f.write("end simulation\n")

    return f"python user.py --user-id {user_id} --username {user_name} --app-host {app_hosts} --app-port {app_ports_str} --app-name {app_names} --agent-host {agent_host} --agent-port {agent_port_str} --agent-id {agent_id} --init-file {init_file_name}"

def run_single_experiment(args: Tuple[str, str, int, int, str]) -> None:
    """Run a single experiment with the given parameters.
    
    Args:
        args: Tuple containing (commands_file, exp_id, duration, minimal_action_taken, log_file)
    """
    commands_file, exp_id, duration, minimal_action_taken, log_file = args
    with open(log_file, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            run_experiment_appless(commands_file, exp_id, duration, minimal_action_taken)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", nargs="+", choices=["azure/gpt-4o-mini-240718-13576", \
                                                            "azure/gpt-4.1-mini-250414-13576", \
                                                            "azure/gpt-4.1-nano-250414-13576", \
                                                            "azure/gpt-4.1-250414-13576", \
                                                            "vertex_ai/gemini-2.0-flash-001", \
                                                            "vertex_ai/gemini-2.5-flash", \
                                                            "vertex_ai/gemini-2.5-pro", \
                                                            "vertex_ai/claude-sonnet-4@20250514", \
                                                            "vertex_ai/claude-3-5-haiku@20241022", \
                                                            "gpt-4.1", \
                                                            "gpt-4.1-mini", \
                                                            "gpt-4.1-nano", \
                                                            "gpt-4o-mini"
                                                            ], \
                                                            default=["azure/gpt-4.1-mini-250414-13576", \
                                                                     "azure/gpt-4.1-mini-250414-13576", \
                                                                     "azure/gpt-4.1-mini-250414-13576"])
    parser.add_argument("--duration", type=int, default=240)
    parser.add_argument("--minimal-action-taken", type=int, default=5)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--example_folder", type=str, default="./example_generation")
    parser.add_argument("--simulation_folder", type=str, default="./simulation_results")
    parser.add_argument("--search_mode", action="store_true", help="Search mode does not add datetime to the simulation folder name.")
    parser.add_argument("--appless", action="store_true", help="Run the simulation in appless mode.")
    parser.add_argument("--num_processes", type=int, default=multiprocessing.cpu_count(),
                      help="Number of concurrent processes to use for running experiments")
    parser.add_argument("--sensitive_data_in_memory", action="store_true")
    parser.add_argument("--version_suffix", type=str, default="")
    args = parser.parse_args()

    print("--------------------------------")
    print(args)
    print("--------------------------------")

    assert len(args.model_list) == 3, "Please provide 3 model names."

    version = args.version
    examples_folder = f"{args.example_folder}/example_{version}"
    file_list = os.listdir(examples_folder)
    file_list = [file for file in file_list if file.endswith(".json") and file.startswith("example_")]
    file_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    if not args.search_mode:
        assert args.version_suffix != "", "Please provide a version suffix when running in search mode."
        simulation_folder = f"{args.simulation_folder}/example_{version}_{args.version_suffix}"
    else:
        simulation_folder = f"{args.simulation_folder}/example_{version}"
    os.makedirs(simulation_folder, exist_ok=True)

    # Collect all experiment parameters first
    all_exp_params = []

    for file in tqdm(file_list, desc="Creating simulation folders...", total=len(file_list)):
        with open(os.path.join(examples_folder, file), "r") as f:
            example = json.load(f)
            
            example_id = int(file.split("_")[-1].replace(".json", ""))
            execution_config = deepcopy(example)
            execution_config["example_id"] = example_id

            current_folder = os.path.join(simulation_folder, file.replace(".json", ""))
            os.makedirs(current_folder, exist_ok=True)

            command_list = []

            apps = example["available_applications"]
            app_ports = find_free_ports(len(apps))
            app_info_dict = {app: port for app, port in zip(apps, app_ports)}

            for app, port in app_info_dict.items():
                command_list.append([f"python application.py --host localhost --port {port} --app-name {app} --db-folder logs", f"{app}.log"])

            agent_ports = find_free_ports(3, start_port=max(app_ports) + 1)

            agent_info_dict = {
                "data_sender": {
                    "id": "agent1",
                    "model": args.model_list[0],
                    "user_id": example["simulation_config"]["data_sender_agent"]["concrete_name"].lower().replace(" ", "_"),
                    "port": agent_ports[0],
                    "apps": deepcopy(apps),
                    "contact": get_third_person_intro_and_contact(example["simulation_config"]["data_sender_agent"], apps)
                },
                "data_subject": {
                    "id": "agent2",
                    "model": args.model_list[1],
                    "user_id": example["simulation_config"]["data_subject_agent"]["concrete_name"].lower().replace(" ", "_"),
                    "port": agent_ports[1],
                    "apps": deepcopy(apps),
                    "contact": get_third_person_intro_and_contact(example["simulation_config"]["data_subject_agent"], apps)
                },
                "data_recipient": {
                    "id": "agent3",
                    "model": args.model_list[2],
                    "user_id": example["simulation_config"]["data_recipient_agent"]["concrete_name"].lower().replace(" ", "_"),
                    "port": agent_ports[2],
                    "apps": [example["simulation_config"]["how_to_check_leakage"]["application"]],
                    "contact": get_third_person_intro_and_contact(example["simulation_config"]["data_recipient_agent"], apps) # Not used
                }
            }

            for agent_role in ["data_sender", "data_subject", "data_recipient"]:
                command_list.append([f"python agent.py --host localhost --port {agent_info_dict[agent_role]['port']} --agent-id {agent_info_dict[agent_role]['id']} --user-id {agent_info_dict[agent_role]['user_id']} --deployment-name {agent_info_dict[agent_role]['model']}", f"{agent_info_dict[agent_role]['id']}.log"])

            for agent_role in ["data_sender", "data_subject", "data_recipient"]:
                agent_dict = example["simulation_config"][agent_role + "_agent"]
                user_id = agent_dict["concrete_name"].lower().replace(" ", "_")
                user_name = agent_dict["concrete_name"]
                social_background = agent_dict["social_background"]

                if agent_role == "data_subject":
                    contact = agent_info_dict["data_sender"]["contact"]
                    task = agent_dict["task"]
                elif agent_role == "data_sender":
                    contact = agent_info_dict["data_subject"]["contact"] + " " + get_third_person_intro_and_contact(example["simulation_config"]["data_recipient_agent"], \
                                                                [example["simulation_config"]["how_to_check_leakage"]["application"]])
                    task = agent_dict["task"] + " " + execution_config["simulation_config"][agent_role + "_agent"]["requirement"]
                elif agent_role == "data_recipient":
                    contact = example["simulation_config"]["data_subject_agent"]["public_profile"] + " " + get_third_person_intro_and_contact(example["simulation_config"]["data_sender_agent"], \
                                                                [example["simulation_config"]["how_to_check_leakage"]["application"]])
                    task = agent_dict["task"] + " " + execution_config["simulation_config"][agent_role + "_agent"]["requirement"]

                sensitive_data = agent_dict["sensitive_data"]
                if isinstance(sensitive_data, dict):
                    sensitive_data = f"{sensitive_data['data_type']}: " + " ".join(sensitive_data["content"])

                init_file_name = f"{agent_role}.txt"
                command_list.append([get_user_command(user_id, user_name, social_background, contact, task, \
                                                      init_file_name, agent_info_dict[agent_role]['apps'], agent_role, sensitive_data, app_info_dict, agent_info_dict), f"{agent_role}.log"])

            # If data subject agent and data sender agent have the same concrete name, merge the command from data_subject.txt to data_sender.txt
            if example["simulation_config"]["data_subject_agent"]["concrete_name"] == example["simulation_config"]["data_sender_agent"]["concrete_name"]:
                with open(os.path.join(current_folder, "data_subject.txt"), "r") as f:
                    data_subject_command = f.read().strip()
                    data_subject_command = data_subject_command.split("wait")[0].strip()

                with open(os.path.join(current_folder, "data_sender.txt"), "r") as f:
                    data_sender_command = f.read()

                command_replace = f"wait http://localhost:{agent_info_dict['data_subject']['port']}/get_task_status active_tasks=0 inactive_tasks=1 timeout=60"
                assert command_replace in data_sender_command, "Command to replace not found in data sender command..."
                data_sender_command = data_sender_command.replace(command_replace, data_subject_command)

                os.remove(os.path.join(current_folder, "data_subject.txt"))
                with open(os.path.join(current_folder, "data_sender.txt"), "w") as f:
                    f.write(data_sender_command)
                
                new_command_list = []
                for command, log_file in command_list:
                    if "data_subject" in command:
                        pass
                    elif "data_sender" in command:
                        original_agent_hosts = command.split("--agent-host")[1].strip().split(" ")[0]
                        original_agent_ports = command.split("--agent-port")[1].strip().split(" ")[0]
                        original_agent_ids = command.split("--agent-id")[1].strip().split(" ")[0]

                        combined_agent_hosts = "localhost localhost"
                        combined_agent_ports = f"{agent_info_dict['data_sender']['port']} {agent_info_dict['data_subject']['port']}"
                        combined_agent_ids = f"{agent_info_dict['data_sender']['id']} {agent_info_dict['data_subject']['id']}"
                        new_command_list.append([command.replace(f"--agent-host {original_agent_hosts} --agent-port {original_agent_ports} --agent-id {original_agent_ids}", \
                                                                f"--agent-host {combined_agent_hosts} --agent-port {combined_agent_ports} --agent-id {combined_agent_ids}"), log_file])
                    else:
                        new_command_list.append([command, log_file])

                command_list = new_command_list
            
            with open(os.path.join(current_folder, "commands.json"), "w") as f:
                json.dump(command_list, f, indent=4)
            
            with open(os.path.join(current_folder, "config.json"), "w") as f:
                json.dump(execution_config, f, indent=4)

            # Collect parameters for all runs of this experiment
            if args.appless:
                for exp_id in range(args.num_runs):
                    log_file = os.path.join(current_folder, f"log_{exp_id}.txt")
                    all_exp_params.append((
                        os.path.join(current_folder, "commands.json"),
                        str(exp_id),
                        args.duration,
                        args.minimal_action_taken,
                        log_file  # We keep this for compatibility but don't use it
                    ))
            else:
                raise NotImplementedError("Not implemented for non-appless mode.")

            if args.debug and args.appless:
                if len(all_exp_params) > args.num_processes:
                    break

    # Run all appless experiments in parallel
    if args.appless and all_exp_params:
        print(f"Running {len(all_exp_params)} experiments in parallel using {min(args.num_processes, len(all_exp_params))} processes")
        with multiprocessing.Pool(processes=min(args.num_processes, len(all_exp_params))) as pool:
            list(tqdm(pool.imap(run_single_experiment, all_exp_params), total=len(all_exp_params), desc="Running experiments"))