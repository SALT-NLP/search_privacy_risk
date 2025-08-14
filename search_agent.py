import json
import os
import dotenv
from openai import AzureOpenAI, OpenAI, AsyncAzureOpenAI, AsyncOpenAI
import litellm
from typing import List, Dict
from utils import retry

dotenv.load_dotenv()

class SearchAgent:
    """
    A basic chat agent supporting multi-round conversations with an LLM.

    Features:
    1. Set system message.
    2. Load conversation history from a JSON file.
    3. Save conversation history to a JSON file.
    4. Receive user query, prompt the LLM with history + query,
       and return the assistant response.
    """

    def __init__(self, model: str = "vertex_ai/gemini-2.5-pro", temperature: float = 1.0, provider: str = "azure", budget_tokens: int = 1024):
        # if provider == "azure":
        #     self.openai = AzureOpenAI(
        #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #         api_version=os.getenv("AZURE_API_VERSION"),
        #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        #     )
        #     self.async_openai = AsyncAzureOpenAI(
        #         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        #         api_version=os.getenv("AZURE_API_VERSION"),
        #         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        #     )
        #     self.model = model
        # elif provider == "openai":
        #     self.openai = OpenAI(
        #         api_key=os.getenv("PERSONAL_OAI_API_KEY")
        #     )
        #     self.async_openai = AsyncOpenAI(
        #         api_key=os.getenv("PERSONAL_OAI_API_KEY")
        #     )
        #     self.model = "gpt-4.1"
        # else:
        #     raise ValueError(f"Invalid provider: {provider}")

        self.model = model
        self.temperature = temperature
        self.messages: List[Dict[str, str]] = []
        self.budget_tokens = budget_tokens

    def set_system_message(self, system_message: str) -> None:
        """Set or replace the system prompt at the start of the conversation."""
        # Remove any existing system messages
        self.messages = [m for m in self.messages if m.get("role") != "system"]
        # Insert new system message at the beginning
        self.messages.insert(0, {"role": "system", "content": system_message})

    def load_history(self, filepath: str) -> None:
        """Load conversation history from a JSON file (sync version)."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.messages = data
            else:
                raise ValueError("History file must contain a list of messages")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading history: {e}")

    async def load_history_async(self, filepath: str) -> None:
        """Load conversation history from a JSON file (async version)."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.messages = data
            else:
                raise ValueError("History file must contain a list of messages")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading history: {e}")
    
    def load_history_from_list(self, messages: List[Dict[str, str]]) -> None:
        """Load conversation history from a list of messages."""
        self.messages = messages

    def save_history(self, filepath: str) -> None:
        """Save the current conversation history to a JSON file (sync version)."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving history: {e}")

    async def save_history_async(self, filepath: str) -> None:
        """Save the current conversation history to a JSON file (async version)."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving history: {e}")

    def chat(self, user_query: str) -> str:
        """
        Add a user query to the conversation, call the LLM, store and return its response (sync version).
        """
        # Append user's message
        self.messages.append({"role": "user", "content": user_query})

        # Send to OpenAI ChatCompletion
        # response = self.openai.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature
        # )

        if "gemini" in self.model:
            response = litellm.completion(
                model=self.model,
                messages=self.messages,
                thinking={"type": "enabled", "budget_tokens": self.budget_tokens}
            )
        elif "gpt" in self.model:
            response = litellm.completion(
                model=self.model,
                messages=self.messages
            )
        else:
            raise ValueError(f"Invalid model: {self.model}")

        assistant_message = response.choices[0].message.content

        # Append assistant's reply to history
        self.messages.append({"role": "assistant", "content": assistant_message})

        # Print usage
        print(f"Usage: {response.usage}")
        return assistant_message

    async def chat_async(self, user_query: str) -> str:
        """
        Add a user query to the conversation, call the LLM, store and return its response (async version).
        """
        # Append user's message
        self.messages.append({"role": "user", "content": user_query})

        # Send to OpenAI ChatCompletion
        # response = await self.async_openai.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature
        # )

        if "gemini" in self.model:
            response = await litellm.acompletion(
                model=self.model,
                messages=self.messages,
                thinking={"type": "enabled", "budget_tokens": self.budget_tokens}
            )
        elif "gpt" in self.model:
            response = await litellm.acompletion(
                model=self.model,
                messages=self.messages
            )
        else:
            raise ValueError(f"Invalid model: {self.model}")

        assistant_message = response.choices[0].message.content

        # Append assistant's reply to history
        self.messages.append({"role": "assistant", "content": assistant_message})

        # Print usage
        print(f"Usage: {response.usage}")
        return assistant_message


if __name__ == "__main__":
    # Example usage
    agent = SearchAgent()
    agent.set_system_message("""
You are a helpful assistant. Your goal is to guess a number between 1 and 10.

If the user says "higher", you should guess a higher number.
If the user says "lower", you should guess a lower number.
If the user says "correct", you should stop guessing.
                             
Return your guess in the following format:
<guess><number></guess>

Your initial guess is 5.
""")

    # Load previous history if exists
    # agent.load_history("history.json")

    while True:
        query = input("User: ")
        if query.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        reply = agent.chat(query)
        print(f"Assistant: {reply}\n")

    # Save conversation history on exit
    agent.save_history("history.json")