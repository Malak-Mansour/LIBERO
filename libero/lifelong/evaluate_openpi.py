'''
EVALUATE OPENPI PI0.5 LIBERO CHECKPOINT

conda deactivate
source ../openpi/.venv/bin/activate

python benchmark_scripts/download_libero_datasets.py --use-huggingface
OR
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface


- EVAL ON task ID 3 libero spatial: pick up the black bowl on the cookie box and place it on the plate
    PYTHONPATH=/home/malak/thesis/LIBERO:$PYTHONPATH \
    /home/malak/thesis/openpi/.venv/bin/python libero/lifelong/evaluate_openpi.py \
    --benchmark libero_spatial \
    --task_id 3 \
    --checkpoint_dir /home/malak/thesis/openpi/checkpoints/pi05_libero \
    --num_trials 5 \
    --device cuda:0

- EVAL ON task ID 9 libero goal: "put the wine bottle on the rack"
    PYTHONPATH=/home/malak/thesis/LIBERO:$PYTHONPATH \
    /home/malak/thesis/openpi/.venv/bin/python libero/lifelong/evaluate_openpi.py \
    --benchmark libero_goal \
    --task_id 9 \
    --checkpoint_dir /home/malak/thesis/openpi/checkpoints/pi05_libero \
    --num_trials 5 \
    --device cuda:0


- EVAL ON OOD TASKS:  
    PYTHONPATH=/home/malak/thesis/LIBERO:$PYTHONPATH \
    /home/malak/thesis/openpi/.venv/bin/python scripts/generate_custom_init_states.py --num_states 20

    PYTHONWARNINGS=ignore::DeprecationWarning \
    PYTHONPATH=/home/malak/thesis/LIBERO:$PYTHONPATH \
    /home/malak/thesis/openpi/.venv/bin/python libero/lifelong/evaluate_openpi.py \
    --benchmark libero_custom \
    --task_id 0 \
    --checkpoint_dir /home/malak/thesis/openpi/checkpoints/pi05_libero \
    --num_trials 5 \
    --device cuda:0 \
    --prompt "pick up the water bottle and stack it on the white box"
'''

import argparse
import collections
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

# Ensure repo root is before this script's directory on sys.path so "datasets" resolves to HF package.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if sys.path and sys.path[0] == str(SCRIPT_DIR):
    sys.path.pop(0)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.video_utils import VideoWriter
from openpi.policies import policy_config
from openpi.training import config as openpi_config
from openpi_client import image_tools


BENCHMARK_MAP = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "libero_custom": "LIBERO_CUSTOM",
}


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    # Copied from robosuite transform_utils; converts wxyz quat to axis-angle.
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def preprocess_images(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Training data was rendered at 256 and rotated 180deg before resize/pad to 224.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    return img, wrist


def build_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def run_episode(env, policy, prompt: str, init_state: np.ndarray, args) -> tuple[bool, list[np.ndarray]]:
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = collections.deque()
    frames = [obs["agentview_image"][::-1]]  # seed video
    success = False

    # let physics settle
    for _ in range(args.num_steps_wait):
        obs, _, done, _ = env.step(np.zeros(7))
        frames.append(obs["agentview_image"][::-1])
        if done:
            return True, frames

    for _ in range(args.max_steps):
        img, wrist = preprocess_images(obs, args.resize_size)

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": build_state(obs),
                "prompt": prompt,
            }
            # pi0.5 predicts an action horizon; we replan every few steps.
            actions = policy.infer(element)["actions"]
            if len(actions) < args.replan_steps:
                raise RuntimeError(f"Policy returned only {len(actions)} actions, expected >= {args.replan_steps}")
            action_plan.extend(actions[: args.replan_steps])

        action = action_plan.popleft()
        obs, _, done, _ = env.step(action.tolist())
        frames.append(obs["agentview_image"][::-1])
        if done:
            success = True
            break

    return success, frames


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenPI pi0.5 LIBERO checkpoint.")
    parser.add_argument("--benchmark", type=str, required=True, choices=list(BENCHMARK_MAP.keys()))
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional custom language prompt to feed the policy. Defaults to the task's canonical language.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.expanduser("~/thesis/openpi/checkpoints/pi05_libero"),
        help="Path to the pi0.5 LIBERO checkpoint (assets/params inside).",
    )
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default="openpi_eval_videos")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    benchmark = get_benchmark(BENCHMARK_MAP[args.benchmark])(0)
    task = benchmark.get_task(args.task_id)
    prompt = args.prompt or task.language

    # Load initial states
    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_states_path, weights_only=False)

    # Build env
    env_args = {
        "bddl_file_name": Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(args.seed)

    # Load pi0.5 LIBERO policy (PyTorch weights)
    train_cfg = openpi_config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint_dir,
        default_prompt=prompt,
        pytorch_device=args.device,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    successes = 0
    for trial in trange(args.num_trials, desc="Trials"):
        init_state = init_states[trial % len(init_states)]
        success, frames = run_episode(env, policy, prompt, init_state, args)
        successes += int(success)

        if not args.no_video:
            video_folder = Path(args.save_dir) / f"task{args.task_id}_trial{trial}"
            with VideoWriter(str(video_folder), save_video=True, fps=20) as writer:
                for frame in frames:
                    writer.append_image(frame)

    env.close()
    print(f"Success rate: {successes}/{args.num_trials} = {successes / args.num_trials:.3f}")


if __name__ == "__main__":
    main()
