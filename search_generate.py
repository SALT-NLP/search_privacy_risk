import json
from json import JSONDecodeError
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAIError, RateLimitError
try:
    from openai import InternalServerError
except ImportError:
    InternalServerError = None  # fallback if not available
import os
import argparse
import random
from utils import parse_response, retry, hash_directory
import numpy as np
from search_collect import get_embedding_sync, calculate_similarity
from search_agent import SearchAgent
import importlib
from copy import deepcopy
import gc
import asyncio
import aiohttp
from tqdm import tqdm
from scipy.stats import rankdata
from search_generate_instructions import get_task_instructions

# Initialize both sync and async clients
sync_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

async_openai = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

async def cleanup_aiohttp():
    for obj in gc.get_objects():
        try:
            if isinstance(obj, aiohttp.ClientSession):
                if not obj.closed:
                    print(f"[CLEANUP] Closing aiohttp session: {obj}")
                    await obj.close()
        except Exception:
            pass

def generate_random_function(N, seed):
    """
    Generates a random bijective function f: {1, ..., N} -> {1, ..., N}
    based on a given seed.

    Returns a dictionary where key i maps to f(i).
    """
    random.seed(seed)
    domain = list(range(1, N + 1))
    codomain = domain.copy()
    random.shuffle(codomain)
    return dict(zip(domain, codomain))

class PromptManager:
    _instance = None
    _prompt_functions = None
    _current_version = "v1"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, version="v1"):
        if cls._instance is None:
            cls._instance = cls()
        cls._instance.set_version(version)
        return cls._instance

    def set_version(self, version):
        if version == self._current_version and self._prompt_functions is not None:
            return

        try:
            prompt_module = importlib.import_module(f"search_prompt_{version}")
            get_attack_system_prompt = prompt_module.get_attack_system_prompt
            get_defense_system_prompt = prompt_module.get_defense_system_prompt
            get_attack_rewrite_query = prompt_module.get_attack_rewrite_query
            get_defense_rewrite_query = prompt_module.get_defense_rewrite_query

            self._prompt_functions = {
                "get_attack_system_prompt": get_attack_system_prompt,
                "get_defense_system_prompt": get_defense_system_prompt,
                "get_attack_rewrite_query": get_attack_rewrite_query,
                "get_defense_rewrite_query": get_defense_rewrite_query
            }
            self._current_version = version
        except ImportError:
            print(f"Warning: search_prompt_{version} not found, falling back to default search_prompt")
            from search_prompt_v1 import (
                get_attack_system_prompt,
                get_defense_system_prompt,
                get_attack_rewrite_query,
                get_defense_rewrite_query
            )
            self._prompt_functions = {
                "get_attack_system_prompt": get_attack_system_prompt,
                "get_defense_system_prompt": get_defense_system_prompt,
                "get_attack_rewrite_query": get_attack_rewrite_query,
                "get_defense_rewrite_query": get_defense_rewrite_query
            }
            self._current_version = "v1"

    def get_attack_system_prompt(self, *args, **kwargs):
        return self._prompt_functions["get_attack_system_prompt"](*args, **kwargs)

    def get_defense_system_prompt(self, *args, **kwargs):
        return self._prompt_functions["get_defense_system_prompt"](*args, **kwargs)

    def get_attack_rewrite_query(self, *args, **kwargs):
        return self._prompt_functions["get_attack_rewrite_query"](*args, **kwargs)

    def get_defense_rewrite_query(self, *args, **kwargs):
        return self._prompt_functions["get_defense_rewrite_query"](*args, **kwargs)

# Initialize the prompt manager with default version
prompt_manager = PromptManager.initialize()

# Export the prompt functions for easy access
get_attack_system_prompt = prompt_manager.get_attack_system_prompt
get_defense_system_prompt = prompt_manager.get_defense_system_prompt
get_attack_rewrite_query = prompt_manager.get_attack_rewrite_query
get_defense_rewrite_query = prompt_manager.get_defense_rewrite_query

def set_prompt_version(version):
    """Set the prompt version to use."""
    prompt_manager.set_version(version)

async def select_examples(
    bank: List[Dict[str, Any]],
    config: Dict[str, Any] = None,
    num_examples: int = 5,
    lambda_param: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Based on rank, select up to `num_examples` from `bank` by maximizing MMR (Maximal Marginal Relevance):
        mmr(e) = lambda * Rel(e) - (1 - lambda) * Redundancy(e, S)
    where:
      - Rel(e) = similarity(config, e) * (1 - e["average_leak_score"]), if config provided
               = (1 - e["average_leak_score"]), if no config provided
      - Redundancy(e, S) = max_{s in S} similarity(e, s)
    
    Args:
        bank: list of examples, each a dict with:
            - "embedding": List[float]
            - "average_leak_score": float in [0,1]
            - any other metadata
        config: arbitrary config object (will be JSON‑serialized). If None, only quality scores will be used.
        num_examples: how many examples to pick
        lambda_param: trade‑off between relevance (1.0) and diversity (0.0)
    
    Returns:
        List of selected examples (up to num_examples), in the order chosen.
    """
    M = len(bank)
    # random.shuffle(bank)

    # print('-' * 100)
    # for item in bank:
    #     print(item["config"]["simulation_config"]["data_recipient_agent"]["task"])
    #     print(item["average_leak_score"])
    # print('-' * 100)

    if M <= num_examples:
        print(f"Skipping selection because there are only {M} examples...")
        return bank.copy()

    # 1) Compute raw relevance scores
    if config is None:
        rel_raw = [1.0 - ex.get("average_leak_score", 1.0) for ex in bank]
    else:
        config_str = json.dumps(config, sort_keys=True)
        current_emb = get_embedding_sync(config_str)
        rel_raw = []
        for ex in bank:
            try:
                sim = calculate_similarity(ex["embedding"], current_emb)
                quality = 1.0 - ex.get("average_leak_score", 1.0)
                rel_raw.append(sim * quality)
            except Exception as e:
                print(f"Error for {ex.get('original_name')}: {e}")
                rel_raw.append(0.0)

    # 2) Turn raw rel into percentiles in [0,1]
    rel_pct = rankdata(rel_raw, method="average")
    rel_pct = [x / M for x in rel_pct]
    # print(rel_pct)

    selected = []

    while len(selected) < num_examples:
        # 3) For each candidate, compute current redundancy raw = max sim to selected
        redun_pct = [0.0] * M
        for i in range(M):
            if i in selected:
                continue
            if selected:
                redun_raw = max(
                    calculate_similarity(bank[i]["embedding"], bank[j]["embedding"])
                    for j in selected
                )
            else:
                redun_raw = 0.0
            redun_pct[i] = redun_raw
        
        redun_pct = rankdata(redun_pct, method="average") / M
        # print(redun_pct)

        # 4) Compute MMR over percentiles for all unselected
        best_idx, best_score = None, -float("inf")
        for i in range(M):
            if i in selected:
                continue
            mmr_score = lambda_param * rel_pct[i] - (1 - lambda_param) * redun_pct[i]
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i

        if best_idx is None:
            break
        selected.append(best_idx)
        print(f"selected {len(selected)} with score {best_score:.2f}:\n{bank[best_idx]['config']['simulation_config']['data_recipient_agent']['task']}\n{bank[best_idx]['config']['simulation_config']['data_sender_agent']['task']}")

    return [bank[i] for i in selected]

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, JSONDecodeError, ValueError, AttributeError))
async def fix_task_description(original_task, generated_task):
    if original_task.strip(".").lower() in generated_task.strip(".").lower():
        return generated_task

    fluent_prompt = f"""# Your Objective

Your goal is to fix the following task description so that it starts with "{original_task}".
Do not change other parts of the task description.

## Input Task Description

{generated_task}

## Output Format

Respond with a task description that starts with "{original_task}".
"""
    
    response = await async_openai.chat.completions.create(
        model="gpt-4.1-250414-13576",
        messages=[{"role": "user", "content": fluent_prompt}],
        temperature=0.0
    )
    print(f"Paraphrasing task description:\n{generated_task} \n-> {response.choices[0].message.content}")
    return response.choices[0].message.content.replace('\n', ' ').strip()

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, JSONDecodeError, ValueError, AttributeError))
async def get_attack_new_config(previous_version, config, selected_bank, history, example_id, search_folder, search_agent_model, no_history=False, best_scores_list=[]):
    async def _try_with_provider(provider):
        if previous_version == "v1" or no_history:
            system_prompt = get_attack_system_prompt(config)
            query = get_attack_rewrite_query(json.dumps(config, indent=4), selected_bank, history, best_scores_list)
            agent = SearchAgent(model=search_agent_model, provider=provider)
            agent.set_system_message(system_prompt)
            response = await agent.chat_async(query)
            new_config = parse_response(response)
            await agent.save_history_async(os.path.join(search_folder, f"search_{example_id}.json"))
        else:
            agent = SearchAgent(model=search_agent_model, provider=provider)
            await agent.load_history_async(os.path.join(search_folder, f"search_{example_id}.json"))
            query = get_attack_rewrite_query(json.dumps(config, indent=4), selected_bank, history, best_scores_list)
            response = await agent.chat_async(query)
            new_config = parse_response(response)
            await agent.save_history_async(os.path.join(search_folder, f"search_{example_id}.json"))

        # The following checks are outside the retry logic, as they are not related to provider switching
        if "simulation_config" not in new_config:
            raise ValueError("No simulation config found in the response")

        if "data_recipient_agent" not in new_config["simulation_config"]:
            raise ValueError("No data recipient agent found in the response")
        
        if "task" not in new_config["simulation_config"]["data_recipient_agent"]:
            raise ValueError("No task found in the data recipient agent")

        return new_config

    try:
        try:
            return await _try_with_provider("azure")
        except Exception as e:
            is_internal_error = False
            if InternalServerError is not None and isinstance(e, InternalServerError):
                is_internal_error = True
            elif hasattr(e, 'status_code') and getattr(e, 'status_code', None) == 500:
                is_internal_error = True
            elif 'InternalServerError' in str(type(e)) or '500' in str(e):
                is_internal_error = True
            if is_internal_error:
                print("500 InternalServerError encountered, retrying with provider='openai'.")
                return await _try_with_provider("openai")
            else:
                raise
    except Exception:
        # Let the retry decorator handle further retries
        raise

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, JSONDecodeError, ValueError, AttributeError))
async def get_defense_new_config(previous_version, whole_config, selected_bank, history, search_folder, search_agent_model, no_history=False, best_scores_list=[]):
    async def _try_with_provider(provider):
        if previous_version == "v1" or no_history:
            system_prompt = get_defense_system_prompt(whole_config)
            query = get_defense_rewrite_query(json.dumps(whole_config, indent=4), selected_bank, history, best_scores_list)
            agent = SearchAgent(model=search_agent_model, provider=provider)
            agent.set_system_message(system_prompt)
            response = await agent.chat_async(query)
            new_config = parse_response(response)
            await agent.save_history_async(os.path.join(search_folder, f"search.json"))
        else:
            agent = SearchAgent(model=search_agent_model, provider=provider)
            await agent.load_history_async(os.path.join(search_folder, f"search.json"))
            query = get_defense_rewrite_query(json.dumps(whole_config, indent=4), selected_bank, history, best_scores_list)
            response = await agent.chat_async(query)
            new_config = parse_response(response)
            await agent.save_history_async(os.path.join(search_folder, f"search.json"))

        # The following checks are outside the retry logic, as they are not related to provider switching
        if "simulation_config" not in new_config:
            raise ValueError("No simulation config found in the response")

        if "data_sender_agent" not in new_config["simulation_config"]:
            raise ValueError("No data sender agent found in the response")
        
        if "task" not in new_config["simulation_config"]["data_sender_agent"]:
            raise ValueError("No task found in the data sender agent")

        return new_config

    try:
        try:
            return await _try_with_provider("azure")
        except Exception as e:
            is_internal_error = False
            if InternalServerError is not None and isinstance(e, InternalServerError):
                is_internal_error = True
            elif hasattr(e, 'status_code') and getattr(e, 'status_code', None) == 500:
                is_internal_error = True
            elif 'InternalServerError' in str(type(e)) or '500' in str(e):
                is_internal_error = True
            if is_internal_error:
                print("500 InternalServerError encountered, retrying with provider='openai'.")
                return await _try_with_provider("openai")
            else:
                raise
    except Exception:
        # Let the retry decorator handle further retries
        raise

def get_attack_history(bank, original_name, top_k=None):
    task_2_score = {}
    for item in bank:
        if item["original_name"] == original_name:
            if item["config"]["simulation_config"]["data_recipient_agent"]["task"] not in task_2_score:
                task_2_score[item["config"]["simulation_config"]["data_recipient_agent"]["task"]] = []
            task_2_score[item["config"]["simulation_config"]["data_recipient_agent"]["task"]].append(item["average_leak_score"])
    
    task_2_average_score = {}
    for task in task_2_score:
        task_2_average_score[task] = float(f"{sum(task_2_score[task]) / len(task_2_score[task]):.2f}")
    
    # sort by average score, from high to low
    task_2_average_score = sorted(task_2_average_score.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        # The lowest tasks will be used for history
        task_2_average_score = task_2_average_score[-top_k:]
    
    return json.dumps(task_2_average_score, indent=4)

def get_defense_history(bank, top_k=None):
    task_2_score = {}
    for item in bank:
        if item["config"]["simulation_config"]["data_sender_agent"]["task"] not in task_2_score:
            task_2_score[item["config"]["simulation_config"]["data_sender_agent"]["task"]] = []
        task_2_score[item["config"]["simulation_config"]["data_sender_agent"]["task"]].append(item["average_leak_score"])
    
    task_2_average_score = {}
    for task in task_2_score:
        task_2_average_score[task] = float(f"{sum(task_2_score[task]) / len(task_2_score[task]):.2f}")
    
    # sort by average score, from high to low
    task_2_average_score = sorted(task_2_average_score.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        # The lowest tasks will be used for history
        task_2_average_score = task_2_average_score[-top_k:]
    
    return json.dumps(task_2_average_score, indent=4)

def bank_processing(bank, goal="attack"):
    current_bank = deepcopy(bank)
    random.shuffle(current_bank)

    new_bank = []
    for item in current_bank:
        history = item["history"]
        config = item["config"]
        average_leak_score = item["average_leak_score"]

        for item in history:
            del item["description"]

        new_item = {
            "average_leak_score": average_leak_score,
            "config": {
                "simulation_config": {
                    "data_recipient_agent" if goal == "attack" else "data_sender_agent": {
                        "task": config["simulation_config"][f"data_recipient_agent" if goal == "attack" else "data_sender_agent"]["task"]
                    }
                }
            },
            "history": history
        }
        new_bank.append(new_item)
    
    return new_bank

async def process_attack_example(example_dir, previous_result_folder, new_example_folder, bank, args, pbar):
    current_bank = deepcopy(bank)
    random.shuffle(current_bank)

    example_id = example_dir.split("_")[-1]

    # version_number = int(args.previous_version.replace("v", ""))
    # random_function = generate_random_function(30, version_number)
    # print(f"Random function: {random_function}")
    
    with open(os.path.join(previous_result_folder, example_dir, "config.json"), "r") as f:
        config = json.load(f)

    if args.local_search:
        history_bank = [item for item in current_bank if item["example_id"] == config["example_id"] and item["original_name"] == config["original_name"]]
        print(f"History bank: {len(history_bank)}")
        attack_history = get_attack_history(history_bank, config["original_name"], top_k=args.history_top_k)

        selected_bank = [item for item in current_bank if item["example_id"] == config["example_id"] and item["original_name"] == config["original_name"] and item["version"] == args.previous_hash]
        print(f"Selected bank: {len(selected_bank)}")
        selected_bank = await select_examples(selected_bank, None, num_examples=args.num_examples, lambda_param=args.lambda_param)
    else:
        # Grouped non local search
        # history_bank = [item for item in current_bank if random_function[int(item["example_id"])] // 3 == random_function[int(config["example_id"])] // 3 and item["original_name"] == config["original_name"]]
        history_bank = [item for item in current_bank if item["original_name"] == config["original_name"]]
        print(f"History bank: {len(history_bank)}")
        attack_history = get_attack_history(history_bank, config["original_name"], top_k=args.history_top_k)

        # selected_bank = [item for item in current_bank if random_function[int(item["example_id"])] // 3 == random_function[int(config["example_id"])] // 3 and item["original_name"] == config["original_name"] and item["version"] == args.previous_hash]
        selected_bank = [item for item in current_bank if item["original_name"] == config["original_name"] and item["version"] == args.previous_hash]
        print(f"Selected bank: {len(selected_bank)}")
        selected_bank = await select_examples(selected_bank, None, num_examples=args.num_examples, lambda_param=args.lambda_param)

    selected_bank = bank_processing(selected_bank, goal="attack")

    selected_bank = "\n".join([json.dumps(item, indent=4) for item in selected_bank])

    new_config = await get_attack_new_config(args.previous_version, config, selected_bank, attack_history, example_id, args.search_folder, args.search_agent_model, no_history=args.no_history, best_scores_list=args.best_scores_list)
    
    config["simulation_config"]["data_recipient_agent"]["task"] = new_config["simulation_config"]["data_recipient_agent"]["task"].replace('\n', ' ').strip()

    # original_task = config["original_tasks"]["data_recipient_agent"]
    # generated_task = new_config["simulation_config"]["data_recipient_agent"]["task"]
    # new_task = await fix_task_description(original_task, generated_task.replace('\n', ' ').strip())
    # config["simulation_config"]["data_recipient_agent"]["task"] = new_task

    with open(os.path.join(new_example_folder, f"{example_dir}.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    pbar.update(1)

async def process_defense_examples(example_dirs, previous_result_folder, new_example_folder, bank, args, pbar):
    current_bank = deepcopy(bank)
    random.shuffle(current_bank)

    whole_config = {}
    for example_dir in example_dirs:
        with open(os.path.join(previous_result_folder, example_dir, "config.json"), "r") as f:
            config = json.load(f)
        whole_config[example_dir] = config

    print(f"History bank: {len(current_bank)}")
    defense_history = get_defense_history(current_bank, top_k=args.history_top_k)

    selected_bank = [item for item in current_bank if item["version"] == args.previous_hash]
    print(f"Selected bank: {len(selected_bank)}")
    selected_bank = await select_examples(selected_bank, None, num_examples=args.num_examples, lambda_param=args.lambda_param)
    
    selected_bank = bank_processing(selected_bank, goal="defense")

    selected_bank = "\n".join([json.dumps(item, indent=4) for item in selected_bank])

    new_config = await get_defense_new_config(args.previous_version, whole_config, selected_bank, defense_history, args.search_folder, args.search_agent_model, no_history=args.no_history, best_scores_list=args.best_scores_list)
    
    for example_dir in example_dirs:
        with open(os.path.join(previous_result_folder, example_dir, "config.json"), "r") as f:
            config = json.load(f)

        config["simulation_config"]["data_sender_agent"]["task"] = new_config["simulation_config"]["data_sender_agent"]["task"].replace('\n', ' ').strip()

        # original_task = config["original_tasks"]["data_sender_agent"]
        # generated_task = new_config["simulation_config"]["data_sender_agent"]["task"]
        # new_task = await fix_task_description(original_task, generated_task.replace('\n', ' ').strip())
        # config["simulation_config"]["data_sender_agent"]["task"] = new_task

        with open(os.path.join(new_example_folder, f"{example_dir}.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        pbar.update(1)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_folder", type=str, required=True)
    parser.add_argument("--previous_version", type=str, required=True)
    parser.add_argument("--new_version", type=str, required=True)
    parser.add_argument("--goal", type=str, default="attack", choices=["attack", "defense"])
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--lambda_param", type=float, default=0.5)
    parser.add_argument("--local_search", action="store_true")
    parser.add_argument("--previous_hash", type=str, default="")
    parser.add_argument("--no_history", action="store_true")
    parser.add_argument("--history_top_k", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--best_scores_list", type=float, nargs="+", default=[])
    parser.add_argument("--prompt_version", type=str, default="v1", help="Version of prompts to use (e.g., v1, v2, v3)")
    parser.add_argument("--search_agent_model", type=str, default="vertex_ai/gemini-2.5-pro")
    args = parser.parse_args()

    # Initialize prompt manager with the specified version
    set_prompt_version(args.prompt_version)

    print(f"[INFO] search_generate, args: {args}...")

    search_folder = args.search_folder
    previous_version = args.previous_version
    new_version = args.new_version

    if args.previous_hash == "":
        previous_hash = hash_directory(search_folder)
        print(f"[INFO] search_generate, version {previous_version} hash: {previous_hash}")
        args.previous_hash = previous_hash
    else:
        previous_hash = args.previous_hash
        print(f"[INFO] search_generate, version {previous_version} hash: {previous_hash}")
    
    if previous_version == "v0":
        if args.goal == "defense":
            print(f"[INFO] Skipping defense generation for v0...")
        else:
            print(f"[INFO] Generating attack instructions for v0...")
            with open(os.path.join(search_folder, "examples", "example_v1", f"example_1.json"), "r") as f:
                config = json.load(f)

            bank_path = os.path.join(search_folder, "bank.json")
            if os.path.exists(bank_path):
                with open(bank_path, "r") as f:
                    bank = json.load(f)
                bank = json.loads(get_attack_history(bank, config["original_name"], top_k=args.num_tasks))
                bank = [item[0] for item in bank]
            else:
                bank = []

            generated_instructions = get_task_instructions(config, args.search_agent_model, args.num_tasks, bank)
            print_generated_instructions = '\n'.join(generated_instructions)
            print(f"[INFO] Generated instructions:\n{print_generated_instructions}")

            for idx, instruction in enumerate(generated_instructions):
                config["simulation_config"]["data_recipient_agent"]["task"] = instruction
                with open(os.path.join(search_folder, "examples", "example_v1", f"example_{idx + 1}.json"), "w") as f:
                    json.dump(config, f, indent=4)
        exit(0)


    bank_path = os.path.join(search_folder, "bank.json")
    if os.path.exists(bank_path):
        with open(bank_path, "r") as f:
            bank = json.load(f)
    else:
        raise ValueError(f"Bank file {bank_path} does not exist")

    previous_result_folder = os.path.join(search_folder, f"results/example_{previous_version}")
    new_example_folder = os.path.join(search_folder, f"examples/example_{new_version}")
    if not os.path.exists(new_example_folder):
        os.makedirs(new_example_folder)

    example_dirs = [item for item in os.listdir(previous_result_folder) 
                   if os.path.isdir(os.path.join(previous_result_folder, item)) 
                   and item.startswith("example_")]
    example_dirs.sort(key=lambda x: int(x.split("_")[-1]))

    # Create progress bar
    pbar = tqdm(total=len(example_dirs), desc="Processing examples")

    if args.goal == "attack":
        # Process attack examples concurrently
        tasks = []
        for example_dir in example_dirs:
            task = process_attack_example(example_dir, previous_result_folder, new_example_folder, bank, args, pbar)
            tasks.append(task)

        # Add timeout and retry logic
        max_retries = 3
        timeout = 180
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
                break
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    print(f"\nTimeout occurred on attempt {attempt + 1}. Retrying...")
                    # Reset progress bar and tasks
                    pbar.n = 0
                    pbar.refresh()
                    tasks = []
                    for example_dir in example_dirs:
                        task = process_attack_example(example_dir, previous_result_folder, new_example_folder, bank, args, pbar)
                        tasks.append(task)
                else:
                    print("\nMaximum retries reached. Some tasks may not have completed.")
                    break
    else:
        # Process defense examples (these are processed sequentially since they share the same task)
        await process_defense_examples(example_dirs, previous_result_folder, new_example_folder, bank, args, pbar)

    pbar.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(cleanup_aiohttp())
