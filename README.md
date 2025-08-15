# Searching for Privacy Risks in LLM Agents via Simulation

**Authors:** Yanzhe Zhang, Diyi Yang

## Abstract

The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. These dynamic dialogues enable adaptive attack strategies that can cause severe privacy violations, yet their evolving nature makes it difficult to anticipate and discover sophisticated vulnerabilities manually. To tackle this problem, we present a search-based framework that alternates between improving attacker and defender instructions by simulating privacy-critical agent interactions. Each simulation involves three roles: data subject, data sender, and data recipient. While the data subject's behavior is fixed, the attacker (data recipient) attempts to extract sensitive information from the defender (data sender) through persistent and interactive exchanges. To explore this interaction space efficiently, our search algorithm employs LLMs as optimizers, using parallel search with multiple threads and cross-thread propagation to analyze simulation trajectories and iteratively propose new instructions. Through this process, we find that attack strategies escalate from simple direct requests to sophisticated multi-turn tactics such as impersonation and consent forgery, while defenses advance from rule-based constraints to identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.

We provide simulation trajectories and search trajectories on [Huggingface](https://huggingface.co/datasets/SALT-NLP/search_privacy_risk).

## Architecture

### Core Components

1. **Agent System** (`agent.py`, `agent_utils.py`)
   - Implements the three-role simulation framework
   - Handles multi-turn conversations and tool usage
   - Manages agent memory and decision-making

2. **Simulation Engine** (`simulation.py`, `launch.py`)
   - Orchestrates multi-agent interactions
   - Manages parallel execution and result collection
   - Provides configurable experiment settings

3. **Search Algorithm** (`search_control.py`, `search_generate.py`)
   - LLM-powered strategy optimization
   - Parallel search with cross-thread propagation
   - Automated instruction generation and refinement

4. **Evaluation Framework** (`evaluation.py`)
   - Privacy leakage detection and scoring
   - Trajectory analysis and pattern recognition
   - Performance metrics and reporting

### Directory Structure

```
├── agent.py                  # Core agent implementation
├── agent_utils.py            # Agent prompts and utilities
├── simulation.py             # Simulation orchestration
├── evaluation.py             # Privacy evaluation framework
├── search_control.py         # Main search algorithm
├── applications/             # Application-specific modules
├── example_generation/       # Training and test examples
├── search/                   # Example search results
└── camel/                    # A copy of the Camel framework, slightly modified
```

## Installation

### Prerequisites

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   # Optional: Add other model provider credentials, Google, Azure, etc
   ```

## Quick Start

### 1. Running a Single Simulation

Here is one command from the search process:

```bash
python simulation.py \
  --model_list gpt-4.1-mini gpt-4.1-mini gpt-4.1-mini \
  --version v1 \
  --num_runs 10 \
  --example_folder ./search/search_test/best/examples \
  --simulation_folder ./search/search_test/best/results \
  --search_mode \
  --appless \
  --num_processes 4
```

**Parameters:**
- `--model_list`: Models for data_subject, data_sender, and data_recipient
- `--version`: Name of the example folder
- `--num_runs`: Number of simulations per configuration
- `--num_processes`: Number of parallel processes

### 2. Evaluating Results

Here is one command from the search process:

```bash
python evaluation.py \
  --example_folder ./search/search_test/best/results/example_v1 \
  --search_mode
```

### 3. Running Full Search Experiments

```bash
python search_control.py \
   --config_dir ./example_generation/example_search_train \
   --output_dir ./example_generation/example_search_train_test \
   --example_ids 16 \
   --search_dir_list search_test \
   --attack_num_examples 5 \
   --num_tasks 30 \
   --num_runs 1 \
   --goal attack \
   --appless \
   --num_processes 30 \
   --max_simulation_round 3 \
   --keep_bank \
   --target_exp_result 0.0 \
   --adaptive_search \
   --prompt_version v1 \
   --no_backtrack \
   --data_sender_model gpt-4.1-mini \
   --data_subject_model gpt-4.1-mini \
   --data_recipient_model gpt-4.1-mini
```

**Key Parameters:**
- `--attack_num_examples`: Number of trajectories for reflection, use `--defense_num_examples` for defense
- `--num_tasks`: Number of threads (default in paper: 30), not used for defense
- `--num_runs`: Number of simulations per configuration (default in paper: 1 for attack, 6 for defense)
- `--num_processes 30`: Number of parallel process used for experiments
- `--goal`: `attack` or `defense` to optimize attacker or defender strategies
- `--max_simulation_round`: Total search steps (default in paper: 10)
- `--adaptive_search`: Enables cross-thread propagation. Use `--local_search_round 1` to turn it off
- `--prompt_version`: Specify the prompt version for LLM optimizers, default in paper is v1

### 4. Tracking Search Progress

Monitor the search process in real-time:

```bash
python show_log_analysis.py test.log
```

Example output:
```
[INFO] Simulation round 1 of 3...
[INFO] Best performance updated: 1.0 -> 0.88
[INFO] Simulation round 2 of 3...
[INFO] Best performance not updated: 0.88 -> 0.95
[INFO] Simulation round 3 of 3...
[INFO] Best performance updated: 0.88 -> 0.35
1.0 -> 0.88 (1) -> 0.88 -> 0.35 (3)
```

### 5. Resampling Examples

Generate additional samples for analysis:

```bash
python search_resample.py \
  --config_dir ./example_generation/example_search_train_test \
  --example_ids 16 \
  --num_runs 20 \
  --appless \
  --num_processes 20 \
  --data_sender_model gpt-4.1-mini \
  --data_subject_model gpt-4.1-mini \
  --data_recipient_model gpt-4.1-mini
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{zhang2025searchingprivacyrisksllm,
      title={Searching for Privacy Risks in LLM Agents via Simulation}, 
      author={Yanzhe Zhang and Diyi Yang},
      year={2025},
      eprint={2508.10880},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2508.10880}, 
}
```

## Acknowledgments

- Built on the [CAMEL framework](https://github.com/camel-ai/camel) for multi-agent systems
- Utilizes [LiteLLM](https://github.com/BerriAI/litellm) for unified model access

## Contact

If you have any questions, please feel free to leave issues or email the first author.
