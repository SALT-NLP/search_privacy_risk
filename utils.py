from typing import Dict, Optional, Any, List
import json
import asyncio
import httpx
import functools
import random
import time
import logging
from datetime import datetime
import re
import os
import ast
import socket
import math
import hashlib

clients_inproc = {}

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def find_free_ports(num_ports, start_port=8000):
    free_ports = []
    current_port = start_port
    while len(free_ports) < num_ports:
        if not is_port_in_use(current_port):
            free_ports.append(current_port)
        current_port += 1
    return free_ports

def get_user_id(name):
    return name.lower().replace(" ", "_")

def get_user_email(name):
    return name.lower().replace(" ", "_") + "@gmail.com"

def second_difference(time_str1, time_str2):
    fmt = "%Y%m%d%H%M%S"
    dt1 = datetime.strptime(time_str1, fmt)
    dt2 = datetime.strptime(time_str2, fmt)
    return abs(int((dt2 - dt1).total_seconds()))

def hash_directory(path, print_it=False):
    hash_obj = hashlib.sha256()
    
    # Get all files sorted by path for consistency
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if ".ipynb_checkpoints" in file_path or "bank" in file_path:
                continue
            files.append(file_path)
    
    files.sort()  # Ensure consistent ordering
    if print_it:
        for file in files:
            print(file)
    
    for filepath in files:
        # Include relative path in hash to detect moved files
        rel_path = os.path.relpath(filepath, path)
        hash_obj.update(rel_path.encode('utf-8'))
        
        # Hash file contents
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
    
    return hash_obj.hexdigest()

def normalize_evaluation_idx(evaluation_idx):
    """
    Normalize the evaluation index to a range of [0, 1).
    Original range: [1, +inf)
    Normalized range: [0, 1)
    """
    logx = math.log(evaluation_idx)
    return float(f"{logx / (logx + 1):.2f}")

def remove_comments(response_content: str) -> str:
    """
    Remove comments that start with // but make sure they're not inside quotes
    """
    in_quotes = False
    quote_char = None
    i = 0
    result = []
    
    while i < len(response_content):
        char = response_content[i]
        
        # Track if we're inside quotes
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and response_content[i-1] != '\\':
                in_quotes = False
            
        # Only remove comments if we're not inside quotes
        if not in_quotes and char == '/' and i+1 < len(response_content) and response_content[i+1] == '/':
            # Skip to the end of line or end of string
            while i < len(response_content) and response_content[i] not in ['\n', '\r']:
                i += 1
        else:
            result.append(char)
            i += 1
            
    response_content = ''.join(result)
    return response_content


def parse_response(response_content: str) -> dict:
    response_content = response_content.strip()

    # 1. Strip code fences
    if "```json" in response_content:
        m = re.search(r'```json(.*?)```', response_content, re.DOTALL)
        response_content = m.group(1) if m else response_content
        # incomplete json block
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].strip()

    # 2. Normalize Python → JSON
    response_content = re.sub(r':\s*None', ': null', response_content)
    response_content = response_content.replace("True", "true").replace("False", "false")
    response_content = response_content.replace("”", '"').replace("“", '"')
    response_content = remove_comments(response_content)  # assume you have this
    response_content = re.sub(r',(\s*[}\]])', r'\1', response_content)

    max_retries = 16
    for attempt in range(1, max_retries + 1):
        try:
            return json.loads(response_content, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"[Attempt {attempt}] JSONDecodeError: {e}")
            logger.error(f"Response content: {response_content}")

            # Count backslashes prior to e.pos to get true index
            prefix = response_content[:e.pos]
            backslash_runs = re.findall(r'\\\\', prefix)
            actual_pos = e.pos - sum(len(bs) // 2 for bs in backslash_runs)

            # Extract the character at or before that position
            # If that char isn't a quote, scan backwards to nearest quote
            idx = actual_pos - 1
            while idx >= 0 and response_content[idx] not in {'"', "'"}:
                idx -= 1
            if idx < 0:
                logger.error("No quote found near error position; aborting retries.")
                break

            # Escape that quote
            logger.info(f"Escaping quote at index {idx}: '{response_content[idx]}'")
            response_content = (
                response_content[:idx] + '\\' + response_content[idx:]
            )

    # Final attempt (or let exception bubble with full context)
    logger.error("Failed to parse JSON after retries. Final content:\n" + response_content)
    return json.loads(response_content, strict=False)

def recover_broken_json(broken_json_str, index_list):
    lines = broken_json_str.strip().splitlines()
    evaluations = []
    current_eval = {}
    field_order = [
        "context_analysis", "context_label",
        "explicit_denial_analysis", "explicit_denial_label",
        "consent_analysis", "consent_label"
    ]
    field_pattern = re.compile(r'^\s*"([^"]+)":\s*(.+),?$')
    index_pointer = 0

    for line in lines:
        match = field_pattern.match(line)
        if match:
            key, value = match.groups()
            key = key.strip()
            value = value.strip().rstrip(',')

            # Remove surrounding quotes from string values
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            current_eval[key] = value

            # When we complete a full group of 6 fields
            if all(f in current_eval for f in field_order):
                if index_pointer >= len(index_list):
                    return broken_json_str

                current_eval["index"] = index_list[index_pointer]
                evaluations.append({
                    "index": current_eval["index"],
                    "context_analysis": current_eval["context_analysis"],
                    "context_label": current_eval["context_label"],
                    "explicit_denial_analysis": current_eval["explicit_denial_analysis"],
                    "explicit_denial_label": current_eval["explicit_denial_label"],
                    "consent_analysis": current_eval["consent_analysis"],
                    "consent_label": current_eval["consent_label"]
                })
                current_eval = {}
                index_pointer += 1

    if index_pointer != len(index_list):
        return broken_json_str

    return json.dumps({"evaluations": evaluations}, indent=2)

def parse_function_call(action_dict: Dict) -> Dict:
    """
    Parses the function call from the action dictionary.
    """
    if "function_call" in action_dict:
        function_call = action_dict["function_call"]
        if function_call is not None:
            if isinstance(function_call["parameters"], dict):
                function_call["parameters"] = {k: v for k, v in function_call["parameters"].items() if v is not None}
            return function_call
        else:
            return None
    else:
        return None

def retry(max_retries=5, initial_delay=1, backoff_factor=2, exceptions=(Exception,), jitter=False):
    """
    A universal retry decorator with increasing delay, supporting both sync and async functions.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_retries:
                        total_delay = delay + random.uniform(0, delay * 0.1) if jitter else delay
                        print(
                            f"Retry {attempt + 1} of {max_retries} after error: {e}. Waiting {total_delay} seconds...")
                        await asyncio.sleep(total_delay)
                        delay *= backoff_factor
                    else:
                        raise  # Re-raise the last exception if max retries exceeded

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_retries:
                        total_delay = delay + random.uniform(0, delay * 0.1) if jitter else delay
                        print(
                            f"Retry {attempt + 1} of {max_retries} after error: {e}. Waiting {total_delay} seconds...")
                        time.sleep(total_delay)
                        delay *= backoff_factor
                    else:
                        raise  # Re-raise the last exception if max retries exceeded

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def get_route_method(route_name: str) -> str:
    """
    Determine the HTTP method (GET or POST) for a given application route name
    based on the verb in the route name.
    
    Args:
        route_name (str): The name of the route to check
        
    Returns:
        str: The HTTP method ('GET' or 'POST')
        
    Raises:
        ValueError: If the route name is not recognized or doesn't follow the expected pattern
    """
    if route_name == "signup":
        return "post"
    elif route_name == "login":
        return "post"

    # Remove leading slash if present
    route_name = route_name.lstrip('/')
    
    # Split the route name into words
    words = route_name.split('_')
    if not words:
        raise ValueError(f"Invalid route name: {route_name}")
        
    verb = words[0].lower()
    
    # Verbs that typically indicate GET requests
    get_verbs = {'get', 'read', 'search', 'list'}
    
    if verb in get_verbs:
        return 'get'
    else:
        return 'post'

async def unified_call(
    base_url: str,
    function_name: str,
    method: Optional[str] = None,
    timeout: float = 60.0,
    **kwargs: Any
) -> Dict:
    """
    Unified function to make HTTP calls to any endpoint (application or agent).
    """
    if method is None:
        # Infer method: get_* -> GET, else POST
        method = get_route_method(function_name)

    url = f"{base_url}/{function_name}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.lower() == "get":
                response = await client.get(url, params=kwargs)
            else:
                response = await client.post(url, params=kwargs)
                
            response.raise_for_status()
            return response.json()
    except httpx.ReadTimeout:
        # Handle timeout specifically with a cleaner error
        logger.warning(f"Unified Call Warning: Request to {url} timed out after {timeout} seconds")
        return {"status": "error", "message": f"Request timed out after {timeout} seconds"}
    except httpx.HTTPStatusError as e:
        # Include response detail if available
        error_detail = ""
        try:
            error_detail = e.response.json()
        except:
            pass
            
        if error_detail and "detail" in error_detail:
            detail_msg = f": {error_detail['detail']}"
        else:
            detail_msg = ""
            
        logger.warning(f"Unified Call Warning: HTTP error {e.response.status_code} for {url}: {e}{detail_msg}")
        return {"status": "error", "message": f"HTTP error: {e.response.status_code}{detail_msg}"}
    except Exception as e:
        logger.warning(f"Unified Call Warning: Error calling {url}: {str(e)}")
        return {"status": "error", "message": str(e)}

def _inproc_request(method: str, url: str, **kwargs):
    """
    Generic in‑process dispatcher.  
    method: 'get', 'post', etc.
    url:   e.g. "http://localhost:8001/send_message"
    kwargs: params=...
    """
    m = re.match(r".+?localhost:(\d+)(/.*)", url)
    if not m:
        raise ValueError(f"URL not recognized for in‑proc: {url}")
    port, path = int(m.group(1)), m.group(2)
    client = clients_inproc.get(port)
    if not client:
        raise ValueError(f"No client for port {port}")
    resp = getattr(client, method.lower())(path, **kwargs)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception as e:
        print(f"Error transforming to JSON: {e}")
        return resp

def get_info_inproc(url: str):
    return _inproc_request("get", url)

def post_info_inproc(url: str, params: dict = None):
    return _inproc_request("post", url, params=params)

def unified_call_inproc(
    base_url: str,
    function_name: str,
    method: str = None,
    timeout: float = 60.0,
    **kwargs: Dict
) -> Dict:
    """
    In-process version of unified_call that routes all HTTP calls to TestClient.
    """
    if method is None:
        method = get_route_method(function_name)

    url = f"{base_url}/{function_name}"
    
    try:
        if method.lower() == "get":
            response = _inproc_request("get", url, params=kwargs)
        else:
            response = _inproc_request("post", url, params=kwargs)
        return response
    except Exception as e:
        print(f"Unified Call Warning: Error calling {url}: {str(e)}")
        return {"status": "error", "message": str(e)}

def extract_start_datetime(s):
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
    match = re.match(pattern, s)
    if match:
        timestamp_str = match.group(1)
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        return dt

    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    match = re.match(pattern, s)
    if match:
        timestamp_str = match.group(1)
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt

    return None

def parse_result(result_str):
    """
    Transform the 'result' string into either None or a Python dict.
    
    Args:
        result_str (str): The result value from the JSON.
        
    Returns:
        dict or None: Parsed dict if it's a valid Python literal dict, otherwise None.
    """
    if not result_str or result_str.strip() in ("None", "null"):
        return None
    try:
        # Safely evaluate into Python objects (dict, list, etc.)
        parsed = ast.literal_eval(result_str)
        if isinstance(parsed, dict):
            return parsed
        return None  # Only keep dicts, discard others
    except (ValueError, SyntaxError):
        return None

def parse_tool_call(tool_call_lines):
    try:
        tool_call_name = None
        for line_idx in range(len(tool_call_lines)):
            if "'external_tool_call_requests': [ToolCallRequest(tool_name=" in tool_call_lines[line_idx][1]:
                tool_call_name = tool_call_lines[line_idx][1].split("'external_tool_call_requests': [ToolCallRequest(tool_name=")[1].split(",")[0].strip().strip("'")
                args = tool_call_lines[line_idx][1].split("args=")[1].split(", tool_call_id")[0].strip()
                args = ast.literal_eval(args)
                break

        if tool_call_name is None:
            return {
                "tool_call_name": None,
                "contains_activity": False,
                "recipient_id": None,
                "args": None,
                "result": None
            }

        result = None
        for line_idx in range(len(tool_call_lines)):
            if tool_call_lines[line_idx][1].startswith("Result: "):
                result = parse_result(tool_call_lines[line_idx][1].split("Result: ")[1].strip())
        

        if tool_call_name == "Messenger_send_message" and "recipient_id" in args:
            contains_activity = True
            recipient_id = args["recipient_id"]
        elif tool_call_name == "Gmail_send_email" and "to_address" in args:
            contains_activity = True
            recipient_id = args["to_address"].split("@")[0]
        elif tool_call_name == "Facebook_create_post":
            contains_activity = True
            recipient_id = "all"
        elif tool_call_name == "Notion_share_page" and "recipient_id" in args:
            contains_activity = True
            recipient_id = args["recipient_id"]
        else:
            contains_activity = False
            recipient_id = None

        return {
            "tool_call_name": tool_call_name,
            "contains_activity": contains_activity,
            "recipient_id": recipient_id,
            "args": args,
            "result": result
        }
    except IndexError:
        print("--------------------------------")
        print("Wrong tool call format:")
        print(tool_call_lines)
        print("--------------------------------")
        return {
            "tool_call_name": None,
            "contains_activity": False,
            "recipient_id": None,
            "args": None,
            "result": None
        }

def parse_cycle(cycle_lines, agent_id, trigger):
    cycle_start_time = cycle_lines[0][0]
    cycle_end_time = cycle_lines[-1][0]
    
    in_tool_call = False
    tool_calls = []
    current_tool_call = []
    for time, line in cycle_lines:
        if line.startswith("Response: "):
            in_tool_call = True
        
        if in_tool_call:
            current_tool_call.append((time, line))

        if line.startswith("Result: "):
            in_tool_call = False
            tool_calls.append(current_tool_call)
            current_tool_call = []

    parsed_tool_calls = [parse_tool_call(tool_call) for tool_call in tool_calls if len(tool_call) > 0]
    contains_activity = any(tool_call["contains_activity"] for tool_call in parsed_tool_calls)
    recipient_ids = [tool_call["recipient_id"] for tool_call in parsed_tool_calls if tool_call["contains_activity"]]

    return {
        "agent_id": agent_id,
        "start_time": cycle_start_time.strftime("%Y%m%d%H%M%S"),
        "end_time": cycle_end_time.strftime("%Y%m%d%H%M%S"),
        "contains_activity": contains_activity,
        "recipient_ids": recipient_ids,
        "trigger": trigger,
        "tool_calls": parsed_tool_calls
    }

def parse_log_file(log_content, agent_id):
    lines = log_content.split("\n")
    time_annotated_lines = []
    for line in lines:
        dt = extract_start_datetime(line)
        if dt is not None:
            time_annotated_lines.append((dt, line.split(" - INFO - ")[-1].strip()))
    
    in_cycle = False
    cycles = []
    current_cycle = []
    for idx, (time, line) in enumerate(time_annotated_lines):
        if line.startswith("Tools:"):
            in_cycle = True

            if time_annotated_lines[idx - 1][1].startswith("New activity detected"):
                trigger = "Notification"
            elif time_annotated_lines[idx - 1][1].startswith("Response timeout reached"):
                trigger = "Timeout"
            else:
                trigger = "Initial"

        if in_cycle:
            current_cycle.append((time, line))

        if line.endswith("actions taken"):
            in_cycle = False
            cycles.append(parse_cycle(current_cycle, agent_id, trigger))
            current_cycle = []
    return cycles

def load_action_cycles(log_dir):
    commands_path = os.path.join("/".join(log_dir.split("/")[:-1]), "commands.json")
    with open(commands_path, "r") as f:
        commands = json.load(f)
    
    data_sender_agent_id = []
    data_sender_user_id = []
    data_recipient_agent_id = []
    data_recipient_user_id = []
    for command_idx, command in enumerate(commands):
        if "data_sender" in command[0]:
            current_agent_id = command[0].split("agent-id")[1].strip().split(" ")[0]
            data_sender_agent_id.append(current_agent_id)
            current_user_id = command[0].split("user-id")[1].strip().split(" ")[0]
            data_sender_user_id.append(current_user_id)
        elif "data_recipient" in command[0]:
            current_agent_id = command[0].split("agent-id")[1].strip().split(" ")[0]
            data_recipient_agent_id.append(current_agent_id)
            current_user_id = command[0].split("user-id")[1].strip().split(" ")[0]
            data_recipient_user_id.append(current_user_id)

    combined_parsed_cycles = []
    for agent_id, user_id in zip(data_sender_agent_id, data_sender_user_id):
        with open(os.path.join(log_dir, f"{agent_id}.log"), "r") as f:
            log_content = f.read()
            parsed_cycles = parse_log_file(log_content, user_id)
            combined_parsed_cycles.extend(parsed_cycles)
    
    for agent_id, user_id in zip(data_recipient_agent_id, data_recipient_user_id):
        with open(os.path.join(log_dir, f"{agent_id}.log"), "r") as f:
            log_content = f.read()
            parsed_cycles = parse_log_file(log_content, user_id)
            combined_parsed_cycles.extend(parsed_cycles)

    # sort by start_time
    combined_parsed_cycles.sort(key=lambda x: x["start_time"])

    return combined_parsed_cycles

def ensure_microseconds(ts: str) -> str:
    """Ensure timestamp has microseconds (6 digits). If missing, append .000000.
    If present but not 6 digits, pad or trim to 6.
    """
    import re
    ts = ts.strip()
    m = re.match(r"^(.*\d{2}:\d{2}:\d{2})(\.(\d{1,6}))?$", ts)
    if not m:
        return ts  # fallback: leave unchanged if pattern unexpected
    base = m.group(1)
    frac = m.group(3) or ""
    frac = (frac + "000000")[:6]
    return f"{base}.{frac}"


def transform_to_activity_log(item):
    """
    Transform an email, message, or post item into the corresponding activity log format
    (matching the screenshots). Ensures `timestamp` has microseconds ('.000000' if absent).
    """
    # Email case
    if "from_address" in item and "to_address" in item:
        return {
            "id": item["id"],
            "description": f"New email, email_id: {item['id']}",
            "app_name": "Gmail",
            "action": "send_email",
            "timestamp": ensure_microseconds(item["time"]),
            "user_id": item["from_address"].split("@")[0],
            "involved_user_ids": [item["to_address"].split("@")[0]],
            "details": {
                "to_address": item["to_address"],
                "subject": item.get("subject", ""),
                "body": item.get("body", ""),
                "cc_address": item.get("cc_address", ""),
                "bcc_address": item.get("bcc_address", ""),
            },
        }
    # Message case
    if "sender_id" in item and "recipient_id" in item:
        return {
            "id": item["id"],
            "description": f"New message, message_id: {item['id']}",
            "app_name": "Messenger",
            "action": "send_message",
            "timestamp": ensure_microseconds(item["time"]),
            "user_id": item["sender_id"],
            "involved_user_ids": [item["recipient_id"]],
            "details": {
                "recipient_id": item["recipient_id"],
                "message": item.get("message", ""),
            },
        }
    # Post case
    if "sender" in item and "content" in item:
        return {
            "id": item["id"],
            "description": f"New post, post_id: {item['id']}",
            "app_name": "Facebook",
            "action": "create_post",
            "timestamp": ensure_microseconds(item["time"]),
            "user_id": item["sender"],
            "involved_user_ids": ["all"],
            "details": {
                "content": item["content"],
            },
        }
    print(item)
    raise ValueError("Unknown item type")

def read_activities(db_dir: str, return_action_cycles=False) -> List[str]:
    """
    Read activities from JSON databases in the given directory.
    """

    activities = []
    
    for file in os.listdir(db_dir):
        if file in ("Gmail_db.json", "Messenger_db.json", "Facebook_db.json"):
            with open(os.path.join(db_dir, file), 'r') as f:
                db = json.load(f)
    
                # If activity_logs present and non-empty, use directly
                if "activity_logs" in db and db["activity_logs"]:
                    activities.extend(db["activity_logs"])
                else:
                    # Fall back to messages, posts, or emails
                    if db.get("messages"):
                        for item in db["messages"].values():
                            activities.append(transform_to_activity_log(item))
                    if db.get("posts"):
                        for item in db["posts"].values():
                            activities.append(transform_to_activity_log(item))
                    if db.get("emails"):
                        for item in db["emails"].values():
                            activities.append(transform_to_activity_log(item))

    activities.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S.%f") if "." in x["timestamp"] else datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))

    action_cycles = load_action_cycles(db_dir)

    for activity_idx, activity in enumerate(activities):
        for cycle_idx, cycle in enumerate(action_cycles):
            if cycle["start_time"] <= activity["timestamp"].split(".")[0].replace(" ", "").replace("-", "").replace(":", "") <= cycle["end_time"]:
                if cycle["agent_id"] == activity["user_id"]:
                    activity["cycle_idx"] = cycle_idx
                    break
        if "cycle_idx" not in activity:
            activity["cycle_idx"] = None

    # print(json.dumps(action_cycles, indent=4))
    # print(json.dumps(activities, indent=4))
    if return_action_cycles:
        return action_cycles

    extended_activities = []
    for activity in activities:
        extended_activities.append(activity)
        
        current_activity_cycle_idx = activity["cycle_idx"]
        current_involved_user_ids = activity["involved_user_ids"]

        next_timeout_activity_cycle_idx = None
        if activity["cycle_idx"] is not None:
            if action_cycles[current_activity_cycle_idx]["contains_activity"]:
                for cycle_idx in range(current_activity_cycle_idx + 1, len(action_cycles)):
                    if action_cycles[cycle_idx]["trigger"] == "Notification" \
                        and action_cycles[cycle_idx]["agent_id"] == activity["user_id"]:
                        break

                    if action_cycles[cycle_idx]["trigger"] == "Timeout" \
                        and action_cycles[cycle_idx]["agent_id"] == activity["user_id"]:
                        next_timeout_activity_cycle_idx = cycle_idx
                        break

        if current_activity_cycle_idx is not None and next_timeout_activity_cycle_idx is not None:
            if current_activity_cycle_idx + 1 < next_timeout_activity_cycle_idx:
                for cycle_idx in range(current_activity_cycle_idx + 1, next_timeout_activity_cycle_idx):
                    user_id = action_cycles[cycle_idx]["agent_id"]
                    reachout_user_id = action_cycles[current_activity_cycle_idx]["agent_id"]
                    reachout_application = activity["app_name"]
                    cycle_contains_activity = action_cycles[cycle_idx]["contains_activity"]
                    if user_id in current_involved_user_ids or "all" in current_involved_user_ids:
                        if not cycle_contains_activity:
                            print("Added activity: ", db_dir)
                            extended_activities.append(
                                {
                                    "description": f"No response to {reachout_user_id} on {reachout_application}",
                                    "cycle_idx": cycle_idx,
                                    "user_id": user_id,
                                }
                            )

    formatted_activities = []
    for i, activity in enumerate(extended_activities):
        if "timestamp" in activity:
            del activity["timestamp"]
        if "id" in activity:
            del activity["id"]
        if "involved_user_ids" in activity:
            del activity["involved_user_ids"]
        if "cycle_idx" in activity:
            del activity["cycle_idx"]
        activity["index"] = i + 1
        formatted_activities.append(activity)
    return formatted_activities


from math import sqrt

def binary_correlation(x, y):
    """
    Compute the phi coefficient (Pearson correlation) between two binary lists x and y.
    
    Args:
        x (List[int]): First list of 0/1 values.
        y (List[int]): Second list of 0/1 values.
        
    Returns:
        float: Correlation in [-1, 1]. Zero if one of the variances is zero.
        
    Raises:
        ValueError: If x and y have different lengths or contain values other than 0 or 1.
    """
    if len(x) != len(y):
        raise ValueError("Lists must have the same length")
    
    # counts for each combination
    n11 = n10 = n01 = n00 = 0
    for xi, yi in zip(x, y):
        if xi not in (0,1) or yi not in (0,1):
            raise ValueError("Lists must contain only 0 or 1")
        if   xi == 1 and yi == 1: n11 += 1
        elif xi == 1 and yi == 0: n10 += 1
        elif xi == 0 and yi == 1: n01 += 1
        else:                     n00 += 1
    
    # marginal totals
    n1_ = n11 + n10
    n0_ = n01 + n00
    n_1 = n11 + n01
    n_0 = n10 + n00
    
    # denominator
    denom = sqrt(n1_ * n0_ * n_1 * n_0)
    if denom == 0:
        return 0.0  # no variation in one or both lists
    
    # phi coefficient
    return (n11 * n00 - n10 * n01) / denom

async def wait_ready(url: str, timeout=60):
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await client.get(url, timeout=1)
                if r.status_code < 400:
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise TimeoutError(f"{url} not ready after {timeout}s")
    
async def wait_for_condition(url: str, **kwargs):
    timeout = int(kwargs.get("timeout", 120))
    kwargs.pop("timeout")

    keys = kwargs.keys()
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await client.get(url)
                r = r.json()
                if all(str(r[key]) == str(kwargs[key]) for key in keys):
                    print(f"{url}: condition {kwargs} met after {time.time() - t0}s")
                    return
            except Exception:
                pass
            await asyncio.sleep(2)
        raise TimeoutError(f"{url}: condition {kwargs} not met after {timeout}s")

def wait_for_condition_inproc(url: str, **kwargs):
    timeout = int(kwargs.get("timeout", 120))
    kwargs.pop("timeout")

    keys = kwargs.keys()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = get_info_inproc(url)
            if all(str(r[key]) == str(kwargs[key]) for key in keys):
                print(f"{url}: condition {kwargs} met after {time.time() - t0}s")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{url}: condition {kwargs} not met after {timeout}s")

async def get_info(url: str):
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url)
            r = r.json()
            return r
        except Exception:
            return {"info": None}
    
if __name__ == "__main__":
    # activities = read_activities("./simulation_results/example_search_train_resample/results/example_v1/example_16/log_0")
    # print("\n\n".join([json.dumps(activity, indent=4) for activity in activities]))

    activities = read_activities("./simulation_results/example_search_train_resample/results/example_v1/example_16/log_0", True)
    print("\n\n".join([json.dumps(activity, indent=4) for activity in activities]))