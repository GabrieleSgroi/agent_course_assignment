import copy
import os
from functools import cached_property
from dataclasses import dataclass
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from smolagents import Model, Tool, ChatMessage
from tenacity import wait_exponential, stop_after_attempt, retry
from importlib import resources

from gaia_multiagent import prompts
from gaia_multiagent.cfg import GenerationCfg
from gaia_multiagent.utils import VerificationError
import time

@dataclass
class GeminiOutput:
    content: str


class GeminiClient:

    @cached_property
    def client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY", None)
        if api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to use Gemini model.")
        return genai.Client(api_key=api_key)

    def clear_all_files(self) -> None:
        files = self.client.files.list()
        for file in files:
            self.client.files.delete(name=file.name)


class GeminiEngine(GeminiClient, Model):
    def __init__(self,
                 model_id: str,
                 cfg: GenerationCfg = GenerationCfg(),
                 ):
        super().__init__()
        self.cfg = cfg
        self.model_id = model_id

    @staticmethod
    def format_role(message: dict) -> dict:
        message = copy.deepcopy(message)
        if message["role"] == "assistant":
            out = {"role": "model", "parts": [{"text": message["content"][0]["text"]}]}
        elif message["role"] == "user":
            out = {"role": "user", "parts": [{"text": message["content"][0]["text"]}]}
        elif message["role"] == "tool-call":
            out = {"role": "model", "parts": [{"text": message["content"][0]["text"]}]}
        elif message["role"] == "tool-response":
            out = {"role": "user", "parts": [{"text": message["content"][0]["text"]}]}
        else:
            raise RuntimeError(f"Unknown role {message["role"]}")
        return out

    @retry(wait=wait_exponential(multiplier=2, min=10, max=100), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict], stop_sequences=None, **kwargs) -> GeminiOutput:
        if stop_sequences is None:
            stop_sequences = self.cfg.stop_sequences

        messages = copy.deepcopy(messages)
        history = [self.format_role(message) for message in messages[1:-1]]

        chat = self.client.chats.create(model=self.model_id, history=history)
        cfg = GenerateContentConfig(system_instruction=messages[0]["content"][0]["text"],
                                    temperature=self.cfg.temperature,
                                    max_output_tokens=self.cfg.max_tokens,
                                    stop_sequences=stop_sequences)
        response = chat.send_message(messages[-1]["content"][0]["text"], config=cfg)
        return GeminiOutput(content=response.text)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)





class GeminiFileQA(GeminiClient):
    def __init__(self,
                 model_id: str,
                 filepath: str,
                 cfg: GenerationCfg = GenerationCfg(),
                 max_wait_loading: int = 10
                 ):
        self.file_id = self.client.files.upload(file=filepath)
        self.model_id = model_id
        self.cfg = cfg
        self.max_wait_loading = max_wait_loading

    def __call__(self, prompt: str) -> str:
        cfg = GenerateContentConfig(temperature=self.cfg.temperature,
                                    max_output_tokens=self.cfg.max_tokens,
                                    )
        waited = 0
        while self.client.files.get(name=self.file_id.name.split('/')[1]).state != "ACTIVE":
            if waited > self.max_wait_loading:
                raise RuntimeError(f"File {self.file_id} is not ready after {self.max_wait_loading} seconds.")
            waited += 1
            time.sleep(5)
        response = self.client.models.generate_content(model=self.model_id, contents=[self.file_id, prompt], config=cfg)
        return response.text


class GeminiVerifier(GeminiClient):
    instruction= resources.read_text(prompts, "verifier.txt")

    def __init__(self,
                 model_id: str,
                 cfg: GenerationCfg = GenerationCfg(),
                 thinking_budget: int = 4096
                 ):
        self.model_id = model_id
        self.cfg = cfg
        self.thinking_budget = thinking_budget



    def verify(self, final_answer:str, agent_memory: any)-> str:
        execution_trace = agent_memory.get_succinct_steps()
        task = execution_trace[0]["task"]
        execution_trace = execution_trace[1:]
        prompt = f"{self.instruction}\nTask:{task}\nAI agent answer: {final_answer}\nExecution:{execution_trace}."
        cfg = GenerateContentConfig(temperature=self.cfg.temperature,
                                    max_output_tokens=self.cfg.max_tokens,
                                    thinking_config=ThinkingConfig(thinking_budget=self.thinking_budget))
        response = self.client.models.generate_content(model=self.model_id, contents=prompt, config=cfg)
        print("EVALUATION: ", response.text)
        if "[WRONG]" in response.text:
            raise VerificationError(f"It seems you made a mistake. Results of the check: {response.text}")
        else:
            return response.text