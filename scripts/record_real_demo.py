#!/usr/bin/env python3
"""
KEYBOARD TELEOP

Sketch script to record real-robot pick-and-place demos for the bottle-on-box task.

Fill in your robot API calls below (camera access, Cartesian moves, gripper control).
Each episode logs a sequence of:
  - front RGB image
  - wrist RGB image
  - state (eef position + quaternion + gripper width)
  - action sent (as returned by your planner/command function)
Plus a prompt string in a sidecar JSON.

Saved to: real_demos/ep_XXXX.npz and ep_XXXX.json

arrow keys (XY), w/s (Z up/down), a (close), d (open) to action
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROMPT = "put the wine bottle on the rack" #"pick up the wine bottle and stack it on the white storage box"
OUT_DIR = Path("real_demos")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Robot API placeholders — replace these with your own stack
# --------------------------------------------------------------------------- #

class Robot:
    def __init__(self):
        # Initialize connection to your robot, cameras, etc.
        pass

    def get_camera(self, name: str) -> np.ndarray:
        """Return an HxWx3 uint8 image from the named camera (e.g., 'front', 'wrist')."""
        raise NotImplementedError

    def eef_pos(self) -> np.ndarray:
        """Return end-effector position (x, y, z)."""
        raise NotImplementedError

    def eef_quat(self) -> np.ndarray:
        """Return end-effector orientation as quaternion (x, y, z, w)."""
        raise NotImplementedError

    def gripper_width(self) -> float:
        """Return gripper width/opening."""
        raise NotImplementedError

    def send_action(self, action: Any):
        """Send a low-level action/command to the robot."""
        raise NotImplementedError

    # Planning helpers — implement as needed for your controller/planner
    def plan_to_pose(self, pose: np.ndarray, z_offset: float = 0.0) -> Any:
        """Return an action that moves to pose (xyzquat), optionally with a z offset."""
        raise NotImplementedError

    def gripper_close(self) -> Any:
        """Return an action that closes the gripper."""
        raise NotImplementedError

    def gripper_open(self) -> Any:
        """Return an action that opens the gripper."""
        raise NotImplementedError

    # Optional: implement small Cartesian jog given a delta xyz (and maybe yaw).
    def jog_cartesian(self, delta_xyz: np.ndarray) -> Any:
        """Return an action that nudges the end-effector by delta_xyz (in meters)."""
        raise NotImplementedError


def get_obs(robot: Robot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = robot.get_camera("front")
    wrist = robot.get_camera("wrist")
    state = np.concatenate([robot.eef_pos(), robot.eef_quat(), [robot.gripper_width()]])
    return rgb, wrist, state


def run_scripted_episode(robot: Robot, bottle_pose: np.ndarray, box_pose: np.ndarray, sleep_s: float = 0.05) -> List[Dict]:
    """Scripted pick-and-place primitive; uses plan_to_pose/gripper_*."""
    traj = []

    def step(action: Any):
        robot.send_action(action)
        time.sleep(sleep_s)
        rgb, wrist, state = get_obs(robot)
        traj.append({"rgb": rgb, "wrist": wrist, "state": state, "action": action})

    # Simple scripted primitive: above bottle -> descend -> close -> lift -> above box -> descend -> open
    step(robot.plan_to_pose(bottle_pose, z_offset=0.10))
    step(robot.plan_to_pose(bottle_pose, z_offset=0.02))
    step(robot.gripper_close())
    step(robot.plan_to_pose(bottle_pose, z_offset=0.10))
    step(robot.plan_to_pose(box_pose, z_offset=0.10))
    step(robot.plan_to_pose(box_pose, z_offset=0.02))
    step(robot.gripper_open())

    return traj


def run_keyboard_episode(robot: Robot, step_m: float = 0.01, sleep_s: float = 0.05) -> List[Dict]:
    """
    Keyboard teleop loop: arrow keys jog XY, W/S jog Z, A closes gripper, D opens.
    Requires Robot.jog_cartesian and gripper_* implementations.
    """
    try:
        import curses
    except ImportError:
        raise ImportError("curses not available; keyboard teleop requires curses.")

    traj: List[Dict] = []

    def step(action: Any):
        robot.send_action(action)
        time.sleep(sleep_s)
        rgb, wrist, state = get_obs(robot)
        traj.append({"rgb": rgb, "wrist": wrist, "state": state, "action": action})

    def loop(stdscr):
        stdscr.nodelay(True)
        stdscr.addstr(0, 0, "Keyboard teleop: arrows=XY, w/s=Z up/down, a=close, d=open, q=quit")
        while True:
            ch = stdscr.getch()
            if ch == -1:
                time.sleep(0.01)
                continue
            if ch in (ord("q"), ord("Q")):
                break
            if ch == curses.KEY_UP:
                step(robot.jog_cartesian(np.array([0.0, step_m, 0.0])))
            elif ch == curses.KEY_DOWN:
                step(robot.jog_cartesian(np.array([0.0, -step_m, 0.0])))
            elif ch == curses.KEY_LEFT:
                step(robot.jog_cartesian(np.array([-step_m, 0.0, 0.0])))
            elif ch == curses.KEY_RIGHT:
                step(robot.jog_cartesian(np.array([step_m, 0.0, 0.0])))
            elif ch in (ord("w"), ord("W")):
                step(robot.jog_cartesian(np.array([0.0, 0.0, step_m])))
            elif ch in (ord("s"), ord("S")):
                step(robot.jog_cartesian(np.array([0.0, 0.0, -step_m])))
            elif ch in (ord("a"), ord("A")):
                step(robot.gripper_close())
            elif ch in (ord("d"), ord("D")):
                step(robot.gripper_open())
    curses.wrapper(loop)
    return traj


def save_episode(traj: List[Dict], idx: int, prompt: str = PROMPT):
    npz_path = OUT_DIR / f"ep_{idx:04d}.npz"
    data = {k: np.stack([t[k] for t in traj]) for k in ["rgb", "wrist", "state", "action"]}
    np.savez_compressed(npz_path, **data)
    meta = {"prompt": prompt, "len": len(traj)}
    (OUT_DIR / f"ep_{idx:04d}.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {npz_path} (steps={len(traj)})")


def main():
    robot = Robot()  # TODO: replace with your robot initialization

    # Choose mode: "scripted" uses plan_to_pose/gripper, "keyboard" uses jogs/gripper.
    mode = "keyboard"  # or "scripted"

    if mode == "scripted":
        bottle_pose = np.array([0, 0, 0, 0, 0, 0, 1])  # TODO: set bottle pose (xyz + quat)
        box_pose = np.array([0, 0, 0, 0, 0, 0, 1])     # TODO: set box pose (xyz + quat)
        num_eps = 5
        for i in range(num_eps):
            traj = run_scripted_episode(robot, bottle_pose, box_pose)
            save_episode(traj, i)
    elif mode == "keyboard":
        traj = run_keyboard_episode(robot, step_m=0.01)
        save_episode(traj, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
