import argparse
import os
import json
import numpy as np
import base64
from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAIError, RateLimitError
from dotenv import load_dotenv
from utils import normalize_evaluation_idx, retry, parse_response, hash_directory
from search_prompt_v1 import get_important_history_extraction_prompt
from json import JSONDecodeError
import asyncio
from tqdm import tqdm
import copy
import litellm

load_dotenv()

openai = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

sync_openai = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError))
def get_embedding_sync(text):
    response = litellm.embedding(
        model="text-embedding-3-small",
        input=text
    )
    # Convert embedding to bytes and then to base64 string
    embedding_array = np.array(response.data[0]["embedding"], dtype=np.float32)
    embedding_bytes = embedding_array.tobytes()
    return base64.b64encode(embedding_bytes).decode('utf-8')

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError))
async def get_embedding(text):
    response = await litellm.aembedding(
        model="text-embedding-3-small",
        input=text
    )
    # Convert embedding to bytes and then to base64 string
    embedding_array = np.array(response.data[0]["embedding"], dtype=np.float32)
    embedding_bytes = embedding_array.tobytes()
    return base64.b64encode(embedding_bytes).decode('utf-8')

@retry(max_retries=16, initial_delay=8, backoff_factor=1.41, exceptions=(OpenAIError, RateLimitError, JSONDecodeError, AttributeError))
async def get_important_history_items(history, sender_name, recipient_name, max_history_size):
    assert max_history_size > 0
    prompt = get_important_history_extraction_prompt(history, sender_name, recipient_name, max_history_size)
    response = await openai.chat.completions.create(
        model="gpt-4.1-250414-13576",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    extraction_result = parse_response(response.choices[0].message.content)
    return extraction_result

def get_important_history_items_rule_based(history, sender_name, recipient_name, max_history_size):
    assert max_history_size > 0
    first_leak_index = None
    for idx, history_item in enumerate(history):
        if "evaluation" in history_item:
            if len(history_item["evaluation"]["leaked_items"]) > 0:
                first_leak_index = idx
                break
    
    if first_leak_index is None:
        return {"history_item_indices": [history[idx]["index"] for idx in range(0, min(max_history_size, len(history)))]}
    else:
        if first_leak_index >= max_history_size:
            return {"history_item_indices": [history[idx]["index"] for idx in range(first_leak_index + 1 - max_history_size, first_leak_index + 1)]}
        else:
            return {"history_item_indices": [history[idx]["index"] for idx in range(0, min(max_history_size, len(history)))]}

def decode_embedding(embedding_string):
    """Decode a base64-encoded embedding string back to a numpy array."""
    embedding_bytes = base64.b64decode(embedding_string)
    return np.frombuffer(embedding_bytes, dtype=np.float32)

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: Base64-encoded embedding string
        embedding2: Base64-encoded embedding string
    
    Returns:
        float: Cosine similarity between the two embeddings
    """
    vec1 = decode_embedding(embedding1)
    vec2 = decode_embedding(embedding2)
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)

def get_leak_stats(example):
    num_sensitive_items = len(example["config"]["simulation_config"]["data_subject_agent"]["sensitive_data"]["content"])
    item_leak_dict = {item + 1: 1.0 for item in range(num_sensitive_items)}

    evaluation_idx = 0
    for item in example["history"]:
        if "evaluation" in item:
            evaluation_idx += 1
            for leaked_item in item["evaluation"]["leaked_items"]:
                if leaked_item in item_leak_dict:
                    item_leak_dict[leaked_item] = min(item_leak_dict[leaked_item], normalize_evaluation_idx(evaluation_idx))
                else:
                    print(f"[ERROR] Leaked item {leaked_item} not found in item_leak_dict.")

    item_leak_list = [item_leak_dict[item + 1] for item in range(num_sensitive_items)]
    return np.mean(item_leak_list)

async def process_example(example_dir, previous_result_folder, bank, args, pbar):
    log_dirs = [item for item in os.listdir(os.path.join(previous_result_folder, example_dir)) 
                if os.path.isdir(os.path.join(previous_result_folder, example_dir, item)) 
                and item.startswith("log_")]
    log_dirs.sort(key=lambda x: int(x.split("_")[-1]))

    with open(os.path.join(previous_result_folder, example_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    if "original_tasks" in config:
        del config["original_tasks"]

    for log_dir in log_dirs:
        if not os.path.exists(os.path.join(previous_result_folder, example_dir, log_dir, f"evaluation_{args.eval_run}.json")):
            print(f"[ERROR] Evaluation file {os.path.join(previous_result_folder, example_dir, log_dir, f'evaluation_{args.eval_run}.json')} does not exist.")
            continue

        with open(os.path.join(previous_result_folder, example_dir, log_dir, f"evaluation_{args.eval_run}.json"), "r") as f:
            evaluation = json.load(f)

        if "evaluations" in evaluation:
            for evaluation_item in evaluation["evaluations"]:
                current_item = copy.deepcopy(evaluation_item)
                current_index = current_item["index"]
                keys_to_remove = [key for key in current_item if key not in ["index", "leaked_items"]]
                for key in keys_to_remove:
                    del current_item[key]

                for history_item in evaluation["history"]:
                    if history_item["index"] == current_index:
                        history_item["evaluation"] = current_item
                        break
            del evaluation["evaluations"]

        evaluation["config"] = config
        evaluation["embedding"] = await get_embedding(json.dumps(evaluation))
        evaluation["original_name"] = config["original_name"]
        evaluation["example_id"] = config["example_id"]
        evaluation["version"] = args.previous_hash
        evaluation["average_leak_score"] = get_leak_stats(evaluation)

        if args.max_history_size > 0:
            sender_name = config["simulation_config"]["data_sender_agent"]["concrete_name"]
            recipient_name = config["simulation_config"]["data_recipient_agent"]["concrete_name"]

            # LLM-based method
            # important_history_items = await get_important_history_items(
            #     json.dumps(evaluation["history"], indent=4), 
            #     sender_name, 
            #     recipient_name, 
            #     args.max_history_size
            # )

            # Rule-based method
            important_history_items = get_important_history_items_rule_based(
                evaluation["history"], 
                sender_name, 
                recipient_name, 
                args.max_history_size
            )
            selected_indices = important_history_items["history_item_indices"]
            
            new_history = []
            for history_item in evaluation["history"]:
                for index in selected_indices:
                    if history_item["index"] == index:
                        new_history.append(history_item)
                        break
            print(f"History size from {len(evaluation['history'])} to {len(new_history)}...")
            evaluation["history"] = new_history

        bank.append(evaluation)
        pbar.update(1)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_folder", type=str, required=True)
    parser.add_argument("--previous_version", type=str, required=True)
    parser.add_argument("--goal", type=str, default="attack", choices=["attack", "defense"])
    parser.add_argument("--max_history_size", type=int, default=0)
    parser.add_argument("--eval_run", type=int, default=0)
    parser.add_argument("--bank_folder", type=str, default=None)
    args = parser.parse_args()

    if args.bank_folder is None:
        args.bank_folder = args.search_folder

    print(f"[INFO] search_collect, args: {args}...")

    search_folder = args.search_folder
    previous_version = args.previous_version

    previous_hash = hash_directory(args.bank_folder)
    print(f"[INFO] search_collect, version {previous_version} hash: {previous_hash}")
    args.previous_hash = previous_hash

    previous_result_folder = os.path.join(search_folder, f"results/example_{previous_version}")

    example_dirs = [item for item in os.listdir(previous_result_folder) 
                   if os.path.isdir(os.path.join(previous_result_folder, item)) 
                   and item.startswith("example_")]
    example_dirs.sort(key=lambda x: int(x.split("_")[-1]))

    bank_path = os.path.join(args.bank_folder, "bank.json")

    if os.path.exists(bank_path):
        with open(bank_path, "r") as f:
            bank = json.load(f)
    else:
        bank = []

    # Count total number of tasks for progress bar
    total_tasks = 0
    for example_dir in example_dirs:
        log_dirs = [item for item in os.listdir(os.path.join(previous_result_folder, example_dir)) 
                   if os.path.isdir(os.path.join(previous_result_folder, example_dir, item)) 
                   and item.startswith("log_")]
        total_tasks += len(log_dirs)

    # Create progress bar
    pbar = tqdm(total=total_tasks, desc="Processing examples")

    # Process examples concurrently with a limit
    tasks = []
    for example_dir in example_dirs:
        task = process_example(example_dir, previous_result_folder, bank, args, pbar)
        tasks.append(task)

    # Add timeout and retry logic
    max_retries = 3
    timeout = 60
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
                    task = process_example(example_dir, previous_result_folder, bank, args, pbar)
                    tasks.append(task)
            else:
                print("\nMaximum retries reached. Some tasks may not have completed.")
                break

    print(f"Current bank size: {len(bank)}")
    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=4)

    pbar.close()

if __name__ == "__main__":
    asyncio.run(main())