from smolagents import Tool

from gaia_multiagent.cfg import GenerationCfg
from gaia_multiagent.engines import GeminiFileQA


class FileQA(Tool):
    name: str
    description: str
    inputs: dict
    task_prompt: str
    output_type: str

    def __init__(self, filepath: str, model_id: str, cfg: GenerationCfg = GenerationCfg()):
        super().__init__()
        self.filepath = filepath
        self.model_id = model_id
        self.cfg = cfg
        self.engine = GeminiFileQA(model_id=model_id, cfg=cfg, filepath=filepath)

    def forward(self, question: str) -> str:
        prompt = self.task_prompt.format(question=question)
        ans = self.engine(prompt)
        return ans


class ImageQA(FileQA):
    name = "ImageQA"
    description = "Use this tool to answer questions about the given image."
    inputs = {
        "question": {
            "type": "string",
            "description": "A precise and detailed question to answer about the image. If possible, provide a small plan "
                           "on the steps to follow to reach the conclusion.",
        },

    }

    task_prompt = ("You are an expert assistant answering questions about the provided image. Think step by step "
                   "before giving the final answer.\nquestion: {question}")

    output_type = "string"


class AudioQA(FileQA):
    name = "AudioQA"
    description = "Use this tool to answer questions about the given audio file."
    inputs = {
        "question": {
            "type": "string",
            "description": "A precise and detailed question to answer about the audio. If possible, provide a small plan "
                           "on the steps to follow to reach the conclusion.",
        },

    }

    task_prompt = ("You are an expert assistant answering questions about the provided audio. Think step by step "
                   "before giving the final answer.\nquestion: {question}")

    output_type = "string"


class VideoQA(FileQA):
    name = "VideoQA"
    description = "Use this tool to answer questions about the given video file."
    inputs = {
        "question": {
            "type": "string",
            "description": "A precise and detailed question to answer about the video. If possible, provide a small plan "
                           "on the steps to follow to reach the conclusion.",
        },
    }
    task_prompt = ("You are an expert assistant answering questions about the provided video. Think step by step before"
                   "giving the final answer.\nquestion: {question}")
    output_type = "string"
