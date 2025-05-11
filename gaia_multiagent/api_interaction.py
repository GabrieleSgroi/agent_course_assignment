import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import requests


class TaskType(StrEnum):
    IMAGE = "image"
    TEXTFILE = "textfile"
    VIDEO = "video"
    AUDIO = "audio"
    NOFILE = "nofile"

    @classmethod
    def from_str(cls, filename: str | None) -> Self:
        if filename is None:
            inst = cls("nofile")
        else:
            extension = os.path.splitext(filename)[1]
            if extension in [".csv", ".txt", ".py", ".xlsx"]:
                inst = cls("textfile")
            elif extension in [".jpg", ".png", ".jpeg"]:
                inst = cls("image")
            elif extension in [".mp4", ".avi", ".mov"]:
                inst = cls("video")
            elif extension in [".mp3", ".wav"]:
                inst = cls("audio")
            else:
                raise ValueError(f"Unknown file type: {filename}")
        return inst


@dataclass(frozen=True)
class Task:
    description: str
    task_id: str
    filepath: str | None

    @property
    def file_type(self) -> TaskType:
        return TaskType.from_str(self.filepath)


def fetch_tasks(files_folder: str = "tmp_files", chunk_size: int = 8192) -> list[Task]:
    response = requests.get("https://agents-course-unit4-scoring.hf.space/questions")
    response.raise_for_status()
    tasks_data = response.json()
    os.makedirs(files_folder, exist_ok=True)
    output = []
    for t in tasks_data:
        if len(fname := t["file_name"]) > 0:
            filepath = os.path.join(files_folder, fname)
            tid = t["task_id"]
            url = f"https://agents-course-unit4-scoring.hf.space/files/{tid}"
            fileresponse = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in fileresponse.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
            output.append(Task(description=t['question'], task_id=tid, filepath=filepath))
        else:
            output.append(Task(description=t['question'], task_id=t['task_id'], filepath=None))
    return output