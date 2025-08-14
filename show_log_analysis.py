import re

def parse_performance_updates(log_str):
    lines = log_str.strip().splitlines()
    updates = []
    round_num = None

    for line in lines:
        if "Simulation round" in line:
            match = re.search(r"Simulation round (\d+)", line)
            if match:
                round_num = int(match.group(1))
        elif "Best performance updated" in line:
            match = re.search(r"(\d+\.\d+)\s*->\s*(\d+\.\d+)", line)
            if match and round_num is not None:
                prev_score, new_score = match.groups()
                updates.append(f"{prev_score} -> {new_score} ({round_num})")

    return " -> ".join(updates)

def process_log_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    groups = []
    current_group = []

    for line in lines:
        if "[INFO] Cleaning up subprocesses..." in line or "[DONE] All commands completed successfully." in line:
            if current_group:
                groups.append(current_group)
                current_group = []
        else:
            if "Best performance updated" in line or "Simulation round" in line or "Best performance not updated" in line:
                current_group.append(line)
    
    # Append the last group if not empty
    if current_group:
        groups.append(current_group)

    # Process and print each group
    for i, group in enumerate(groups):
        info_lines = [line for line in group if line.strip().startswith("[INFO]")]
        print(f"\n--- Group {i+1} ---")
        for info_line in info_lines:
            print(info_line.strip())
        
        print(parse_performance_updates("\n".join(group)))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python backtrack_analysis.py <path_to_log_file>")
    else:
        process_log_file(sys.argv[1])