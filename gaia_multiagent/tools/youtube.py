import os

from pytubefix import YouTube
from smolagents import Tool

from gaia_multiagent.cfg import GenerationCfg
from gaia_multiagent.tools.files import FileQA, VideoQA


class YouTubeQA(Tool):
    name = "YouTubeQA"
    description = "Use this tool to answer questions about a YouTube video."
    inputs = {
        "question": {
            "type": "string",
            "description": "A precise and detailed question to answer about the video. If possible, provide a small plan "
                           "on the steps to follow to reach the conclusion."
        },
        "url": {
            "type": "string",
            "description": "The URL of the YouTube video to answer questions about."
        }

    }

    task_prompt = ("You are an expert assistant answering questions about the provided image. Think step by step "
                   "before giving the final answer.\nquestion: {question}")

    output_type = "string"

    def __init__(self, model_id: str, output_dir: str, cfg: GenerationCfg = GenerationCfg()):
        super().__init__()
        self.output_dir = output_dir
        self.cfg = cfg
        self.model_id = model_id

    def download_video(self, url: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filename = url.split('=')[1] + '.mp4'
        YouTube(url).streams.first().download(output_path=self.output_dir, filename=filename)
        return os.path.join(self.output_dir, filename)

    def forward(self, question: str, url: str) -> str:
        filepath = self.download_video(url=url)
        fileqa = VideoQA(filepath=filepath, model_id=self.model_id, cfg=self.cfg)
        return fileqa(question)
