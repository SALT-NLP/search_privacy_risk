from search_control import search_loop
import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--example_ids", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_runs", nargs="+", type=int, required=True)
    parser.add_argument("--goal", type=str, default="defense", choices=["attack", "defense"])
    parser.add_argument("--search_dir_list", nargs="+", type=str, default=None)
    parser.add_argument("--local_search_round", type=int, default=0)
    parser.add_argument("--non_local_search_round", type=int, default=0)
    parser.add_argument("--adaptive_search", action="store_true")
    parser.add_argument("--attack_num_examples", type=int, default=5)
    parser.add_argument("--attack_num_other_examples", type=int, default=0)
    parser.add_argument("--defense_num_examples", type=int, default=5)
    parser.add_argument("--lambda_param", type=float, default=1.0)
    parser.add_argument("--max_simulation_round", type=int, default=1)
    parser.add_argument("--max_history_size", type=int, default=10)
    parser.add_argument("--appless", action="store_true")
    parser.add_argument("--starting_iteration", type=int, default=1)
    parser.add_argument("--retry_generation_limit", type=int, default=2)
    parser.add_argument("--target_exp_result", type=float, default=0.7)
    parser.add_argument("--keep_bank", action="store_true")
    parser.add_argument("--num_processes", type=int, default=10)
    parser.add_argument("--ratio_of_best_runs", type=float, default=1.0)
    parser.add_argument("--no_history", action="store_true")
    parser.add_argument("--history_top_k", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--no_backtrack", action="store_true")
    parser.add_argument("--prompt_version", type=str, default="v1", help="Version of prompts to use (e.g., v1, v2, v3)")
    parser.add_argument("--data_sender_model", type=str, default="azure/gpt-4.1-mini-250414-13576")
    parser.add_argument("--data_subject_model", type=str, default="azure/gpt-4.1-mini-250414-13576")
    parser.add_argument("--data_recipient_model", type=str, default="azure/gpt-4.1-mini-250414-13576")
    parser.add_argument("--sensitive_data_in_memory", action="store_true")
    parser.add_argument("--resample_id", type=str, default=None)
    parser.add_argument("--search_agent_model", type=str, default="vertex_ai/gemini-2.5-pro")
    parser.add_argument("--best_scores_list", type=float, nargs="+", default=[])
    args = parser.parse_args()

    args.adaptive_search = True
    args.keep_bank = True
    args.no_backtrack = True
    args.goal = "defense"

    if args.example_ids is None:
        example_ids = [item for item in os.listdir(args.config_dir) if item.startswith("example_") and item.endswith(".json")]
        example_ids = [int(item.split("_")[-1].split(".")[0]) for item in example_ids]
        example_ids.sort()
        args.example_ids = example_ids

    print(f"[INFO] args: {args}")

    command_template = [
        "python search_generate.py --search_folder ./simulation_results/{search_run} --previous_version v{previous_iteration} --new_version v{iteration} --goal defense --num_examples {defense_num_examples} --lambda_param {lambda_param} --history_top_k {history_top_k} --num_tasks {num_tasks} --prompt_version {prompt_version} --best_scores_list {best_scores_list}",
        "python simulation.py --model_list {data_sender_model} {data_subject_model} {data_recipient_model} --version v{iteration} --num_runs {num_runs} --example_folder ./simulation_results/{search_run}/examples --simulation_folder ./simulation_results/{search_run}/results",
        "python evaluation.py --example_folder ./simulation_results/{search_run}/results/example_v{iteration} --search_mode",
        "evaluate_the_best_run",
        "python search_collect.py --search_folder ./simulation_results/{search_run} --previous_version v{iteration} --goal defense --max_history_size {max_history_size}",
        "checkpoint_and_restore"
    ]
    next_dir_name = args.config_dir.split("/")[-1] + "_resample" + f"_{args.resample_id}" if args.resample_id is not None else args.config_dir.split("/")[-1] + "_resample"
    checkpoint_folder = None

    os.makedirs(os.path.join("./simulation_results", next_dir_name), exist_ok=True)
    os.makedirs(os.path.join("./simulation_results", next_dir_name, "examples"), exist_ok=True)
    os.makedirs(os.path.join("./simulation_results", next_dir_name, "examples", "example_v1"), exist_ok=True)
    os.makedirs(os.path.join("./simulation_results", next_dir_name, "results"), exist_ok=True)

    for example_id in args.example_ids:
        config = json.load(open(os.path.join(args.config_dir, f"example_{example_id}.json"), "r"))
        with open(os.path.join("./simulation_results", next_dir_name, "examples", "example_v1", f"example_{example_id}.json"), "w") as f:
            json.dump(config, f, indent=4)

    search_loop(checkpoint_folder, next_dir_name, command_template, args)