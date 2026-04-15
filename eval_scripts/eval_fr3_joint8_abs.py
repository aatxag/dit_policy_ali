#!/usr/bin/env python3
"""
Eval script for dit-policy with ABSOLUTE ACTIONS in joint space (8D).

Control mode: joint impedance (same as GELLO teleoperation).
  tau = Kp*(q_goal - q_actual) + Kd*(-dq_filtered)

The policy publishes a new target joint position at ~10 Hz.
The impedance controller tracks it at 1 kHz via torques — no trajectory
planning, no joint_motion_generator, no velocity/acceleration constraints.
"""

import argparse
import json
import os
from collections import deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml

from fr3_eval_env import make_fr3_env

torch.set_float32_matmul_precision("high")

FR3_JOINT_LIMITS_LOW = np.array(
    [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895, 0.0],
    dtype=np.float32,
)
FR3_JOINT_LIMITS_HIGH = np.array(
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895, 0.08],
    dtype=np.float32,
)


class Policy:
    def __init__(self, agent_path, model_name, args):
        self.args = args

        with open(Path(agent_path, "agent_config.yaml"), "r", encoding="utf-8") as f:
            agent_config = yaml.safe_load(f.read())
        with open(Path(agent_path, "obs_config.yaml"), "r", encoding="utf-8") as f:
            obs_config = yaml.safe_load(f.read())
        with open(Path(agent_path, "ac_norm.json"), "r", encoding="utf-8") as f:
            ac_norm = json.load(f)
            self.loc   = np.array(ac_norm["loc"],   dtype=np.float32)
            self.scale = np.array(ac_norm["scale"], dtype=np.float32)

        print(f"[INFO] ac_norm loc:   {np.round(self.loc, 4)}")
        print(f"[INFO] ac_norm scale: {np.round(self.scale, 4)}")
        if self.scale.mean() < 0.01:
            print("[WARN] scale very small — are these delta actions instead of absolute?")

        agent = hydra.utils.instantiate(agent_config)
        with torch.serialization.safe_globals(["omegaconf.listconfig.ListConfig"]):
            save_dict = torch.load(
                Path(agent_path, model_name), map_location="cpu", weights_only=False
            )
        agent.load_state_dict(save_dict["model"])
        self.agent = torch.compile(agent.eval().cuda().get_actions)

        self.transform    = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys     = obs_config["imgs"]
        self.pred_horizon = args.pred_horizon

        print(f"[INFO] Loaded:     {agent_path}/{model_name}")
        print(f"[INFO] Step:       {save_dict.get('global_step', 'unknown')}")
        print(f"[INFO] Img keys:   {self.img_keys}")
        print("[INFO] Action:     ABSOLUTE joint positions")

        self.reset()

    def reset(self):
        self.act_history = deque(maxlen=self.pred_horizon)
        self.last_ac     = None

    def _proc_images(self, img_dict, size=(256, 256)):
        torch_imgs = {}
        for i, k in enumerate(self.img_keys):
            if k not in img_dict:
                raise KeyError(f"Key '{k}' not found. Available: {list(img_dict.keys())}")
            bgr = cv2.resize(img_dict[k][:, :, :3], size, interpolation=cv2.INTER_AREA)
            rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
            torch_imgs[f"cam{i}"] = self.transform(rgb)[None].cuda()[None]
        return torch_imgs

    def _infer(self, obs):
        img   = self._proc_images(obs["images"])
        state = torch.from_numpy(obs["qpos"]).float()[None].cuda()
        with torch.no_grad():
            ac = self.agent(img, state)[0].detach().cpu().numpy().astype(np.float32)
        return ac[: self.pred_horizon]   # (N, 8) normalized

    def forward(self, obs) -> np.ndarray:
        """
        Return next (8,) absolute target position.
        Uses chunked execution with gamma smoothing between chunks.
        """
        if not len(self.act_history):
            for ac in self._infer(obs):
                self.act_history.append(ac)

        raw_ac  = self.act_history.popleft()
        last_ac = self.last_ac if self.last_ac is not None else raw_ac
        self.last_ac = self.args.gamma * raw_ac + (1.0 - self.args.gamma) * last_ac

        # Denormalize
        target = self.last_ac * self.scale + self.loc
        target = np.clip(target, FR3_JOINT_LIMITS_LOW, FR3_JOINT_LIMITS_HIGH)

        # Delta limit relative to current robot position
        current = obs["qpos"]
        delta   = target - current
        delta[:7] = np.clip(delta[:7], -self.args.dq_limit, self.args.dq_limit)
        delta[7:] = np.clip(delta[7:], -self.args.dg_limit, self.args.dg_limit)

        return current + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)

    parser.add_argument("--T",            default=300,  type=int)
    parser.add_argument("--num_rollouts", default=1,    type=int)
    parser.add_argument("--pred_horizon", default=8,    type=int)
    parser.add_argument("--gamma",        default=0.85, type=float)

    parser.add_argument("--hz",       default=10.0, type=float)
    parser.add_argument("--dq_limit", default=0.15, type=float,
                        help="Max joint delta per step (rad). "
                             "Impedance control tracks this safely via PD torques.")
    parser.add_argument("--dg_limit", default=0.02, type=float)

    parser.add_argument("--settle_time", default=1.0, type=float)

    parser.add_argument("--joint_topic",
                        default="/joint_states", type=str)
    parser.add_argument("--image_topic",
                        default="/camera/camera_wrist/color/image_raw", type=str)
    parser.add_argument("--impedance_topic",
                        default="/gello/joint_states", type=str,
                        help="Topic the joint impedance controller subscribes to.")

    parser.add_argument("--open_width",            default=0.08, type=float)
    parser.add_argument("--close_width",           default=0.01, type=float)
    parser.add_argument("--gripper_speed",         default=0.05, type=float)
    parser.add_argument("--gripper_force",         default=60.0, type=float)
    parser.add_argument("--gripper_epsilon_inner", default=0.01, type=float)
    parser.add_argument("--gripper_epsilon_outer", default=0.01, type=float)

    args = parser.parse_args()
    args.period = 1.0 / args.hz

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)

    policy = Policy(agent_path, model_name, args)

    env = make_fr3_env(
        init_node=True,
        hz=args.hz,
        dq_limit=args.dq_limit,
        dg_limit=args.dg_limit,
        settle_time=args.settle_time,
        joint_topic=args.joint_topic,
        image_topic=args.image_topic,
        impedance_topic=args.impedance_topic,
        open_width=args.open_width,
        close_width=args.close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_epsilon_inner=args.gripper_epsilon_inner,
        gripper_epsilon_outer=args.gripper_epsilon_outer,
    )

    for rollout_num in range(args.num_rollouts):
        last_input = None
        while last_input != "y":
            if last_input == "r":
                env.reset()
            last_input = input("Continue with rollout (y; r to reset)? ").strip().lower()

        print(f"[INFO] Starting rollout {rollout_num}")
        policy.reset()
        obs = env.reset()

        for t in range(args.T):
            action = policy.forward(obs.observation)
            print(
                f"[STEP {t:04d}] target={np.round(action[:7], 4)}  "
                f"gripper={action[7]:.4f}"
            )
            obs = env.step(action)

        env.reset_gripper()

    env.shutdown()


if __name__ == "__main__":
    main()
