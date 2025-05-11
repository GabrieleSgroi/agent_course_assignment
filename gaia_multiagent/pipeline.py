from importlib import resources

import pandas as pd
from smolagents import CodeAgent

from gaia_multiagent import prompts
from gaia_multiagent.api_interaction import Task, TaskType
from gaia_multiagent.engines import GeminiEngine, GeminiVerifier
from gaia_multiagent.tools.files import ImageQA, AudioQA
from gaia_multiagent.tools.search import WebSearchAssistant
from gaia_multiagent.utils import InternetSearch, load_as_txt


def multiagent_pipeline(task: Task,
                        engine_model_id: str = "gemini-2.0-flash",
                        verifier_model_id: str = "gemini-2.5-flash-preview-04-17") -> tuple[str, dict]:
    engine = GeminiEngine(model_id=engine_model_id)
    question = task.description
    search_assistant_tool = WebSearchAssistant(engine=engine,
                                               search_engine=InternetSearch())
    tools = [search_assistant_tool]
    base_prompt = (f"Find the answer to the following question: {question}. \n"
                   "If you search on the web, don't use the same (or very similar) query twice. Don't search on the web"
                   " for trivial and well known common knowledge.\n"
                   "Your final answer should only contain what is requested (it will be checked with an exact match)"
                   "and nothing else, not even 'Final answer:', other symbols, or final punctuation. "
                   "Numerical answer must be in numbers.")
    if task.file_type == TaskType.IMAGE:
        tools.append(ImageQA(model_id=engine_model_id, filepath=task.filepath))
        base_prompt += "You can use the provided image."
    if task.file_type == TaskType.AUDIO:
        tools.append(AudioQA(model_id=engine_model_id, filepath=task.filepath))
        base_prompt += "You can use the provided audio."
    if task.file_type == TaskType.TEXTFILE:
        file_content = load_as_txt(filepath=task.filepath)
        base_prompt += f"You can use the provided file {task.filepath} whose content is reported below:\n{file_content}"
    verifier = GeminiVerifier(model_id=verifier_model_id)
    manager_agent = CodeAgent(model=engine,
                               tools=tools,
                               planning_interval=3,
                               verbosity_level=2,
                               final_answer_checks=[verifier.verify],
                               additional_authorized_imports=["pandas"],
                               max_steps=15)
    manager_agent.prompt_templates["planning"]["initial_plan"] = resources.read_text(prompts, "initial_planning.txt")
    print("Starting execution...")
    ans = manager_agent.run(base_prompt)
    engine.clear_all_files()
    return ans, manager_agent.memory.get_succinct_steps()



