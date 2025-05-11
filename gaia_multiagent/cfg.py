from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class RetrieverCfg:
    chunk_size: int = 2048
    chunk_overlap: int = 512
    length_function: Callable = len
    k: int = 15
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", "\t", ".", " ", ""])


@dataclass(frozen=True)
class GenerationCfg:
    max_tokens: int = 2048
    temperature: float = 0.
    stop_sequences: list[str] = field(default_factory=lambda: ["END"])


@dataclass(frozen=True)
class SearchAssistantCfg:
    retriever_cfg: RetrieverCfg = RetrieverCfg()
    max_steps: int = 7
    verbosity_level: int = 1
    planning_interval: int = 5
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2" #TODO: CHANGE THIS TO A BETTER EMBEDDING MODEL
