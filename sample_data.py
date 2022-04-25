import shutil, os
from os import path
import random

tasks = [f"task_{idx}" for idx in range(1, 8)]

source_path = "/workspace/conpro_experiments/data/AnimalFace/cl"
target_path = "/workspace/conpro_experiments/data/AnimalFace/cl-10"

n_shots = 10

for task in tasks:
    task_dir = path.join(source_path, task)
    sub_dir = os.listdir(task_dir)
    for name in sub_dir:
        if "Head" in name:
            sub_dir = name
            print(sub_dir)
            break
    target_dir = path.join(task_dir, sub_dir)
    files = os.listdir(target_dir)
    chosen_files = random.sample(files, k=n_shots)
    for f in chosen_files:
        save_dir = path.join(target_path, task, sub_dir)
        if not path.isdir(save_dir):
            os.makedirs(save_dir)
        shutil.copyfile(path.join(target_dir, f), path.join(save_dir, f))
