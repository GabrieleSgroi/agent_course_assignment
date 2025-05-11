import os
from argparse import ArgumentParser

import pandas as pd

from gaia_multiagent.api_interaction import fetch_tasks
from gaia_multiagent.pipeline import multiagent_pipeline


def run_all(save_csv_path:str)->None:
    tasks = fetch_tasks()
    if os.path.exists(save_csv_path):
        df = pd.read_csv(save_csv_path)
        new = False
    else:
        df = None
        new = True
    for t in tasks:
        if (df is not None) and (t.task_id in df["task_id"].tolist()):
            continue
        print("Solving task: ", t.description)
        ans, succint_steps = multiagent_pipeline(task=t,
                                                 engine_model_id="gemini-2.0-flash",
                                                 verifier_model_id="gemini-2.5-flash-preview-04-17")
        if isinstance(ans, str):
            ans = ans.replace(", ", ",").replace(",", ", ") #Sanitize commas
        print("Final answer: ", ans)
        save_df = pd.DataFrame([{"task_id":t.task_id, "submitted_answer":ans, "steps":succint_steps}])
        save_df.to_csv(save_csv_path, mode="a", header=new, index=False)
        new = False

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_csv_path", type=str, default="submit_answers.csv")
    args = parser.parse_args()
    run_all(save_csv_path=args.save_csv_path)

