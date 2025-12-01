#!/usr/bin/env python3
"""
Generate init states for the custom LIBERO task "pick up the water bottle and stack it on the white box".

Usage:
  PYTHONPATH=./libero:$PYTHONPATH python scripts/generate_custom_init_states.py --num_states 20
"""

import argparse
import os
from pathlib import Path

import torch
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_states", type=int, default=20, help="Number of init states to save")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark = get_benchmark("LIBERO_CUSTOM")(0)
    task = benchmark.get_task(0)

    env_args = {
        "bddl_file_name": os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ),
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    init_states = []
    for _ in range(args.num_states):
        env.reset()
        state = env.sim.get_state().flatten()
        init_states.append(state)

    init_states = torch.stack([torch.from_numpy(s) for s in init_states])

    out_dir = Path(get_libero_path("init_states")) / task.problem_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task.name}.init"
    torch.save(init_states, out_path)
    print(f"Saved {len(init_states)} init states to {out_path}")


if __name__ == "__main__":
    main()
