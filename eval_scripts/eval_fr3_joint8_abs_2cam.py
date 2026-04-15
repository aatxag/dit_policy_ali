#!/usr/bin/env python3
"""
Eval script for dit-policy — ABSOLUTE ACTIONS, TWO CAMERAS.

Same as eval_fr3_joint8_abs.py but reads cam0 (wrist) and cam1 (external).
The policy's obs_config.yaml must list both: imgs: [cam0, cam1].

Usage:
  python eval_fr3_joint8_abs_2cam.py path/to/checkpoint.ckpt \
      --image_topic_1 /camera/camera_wrist/color/image_raw \
      --image_topic_2 /camera/camera_d405/color/image_raw
"""

import argparse
import json
import os
import pickle
from collections import deque
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml

from fr3_eval_env_2cam import make_fr3_env_2cam

torch.set_float32_matmul_precision("high")

FR3_JOINT_LIMITS_LOW = np.array(
    [-2.3093, -1.5133, -2.4937, -3.0500, -2.4800, 0.8521, -2.6895, 0.0],
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
        self.img_keys     = obs_config["imgs"]   # expects ["cam0", "cam1"]
        self.pred_horizon = args.pred_horizon

        print(f"[INFO] Loaded:     {agent_path}/{model_name}")
        print(f"[INFO] Step:       {save_dict.get('global_step', 'unknown')}")
        print(f"[INFO] Img keys:   {self.img_keys}")
        print("[INFO] Action:     ABSOLUTE joint positions")
        print("[INFO] Cameras:    2 (wrist + external)")

        if len(self.img_keys) != 2:
            print(f"[WARN] obs_config has {len(self.img_keys)} img key(s): {self.img_keys}. "
                  "Expected 2 for this script.")

        self.reset()

    def reset(self):
        self.act_history = deque(maxlen=self.pred_horizon)
        self.last_ac     = None

    def _proc_images(self, img_dict, size=(256, 256)):
        torch_imgs = {}
        for i, k in enumerate(self.img_keys):
            if k not in img_dict:
                raise KeyError(f"Key '{k}' not found in obs. Available: {list(img_dict.keys())}")
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

    def forward(self, obs, pred_norm=None) -> np.ndarray:
        # Replan every step: infer fresh actions and take only the first one.
        # pred_norm can be passed in to avoid a second inference call.
        if pred_norm is None:
            pred_norm = self._infer(obs)[0]

        last_ac = self.last_ac if self.last_ac is not None else pred_norm
        self.last_ac = self.args.gamma * pred_norm + (1.0 - self.args.gamma) * last_ac

        target = self.last_ac * self.scale + self.loc
        target = np.clip(target, FR3_JOINT_LIMITS_LOW, FR3_JOINT_LIMITS_HIGH)

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
    parser.add_argument("--dq_limit", default=0.15, type=float)
    parser.add_argument("--dg_limit", default=0.02, type=float)

    parser.add_argument("--settle_time", default=1.0, type=float)

    parser.add_argument("--joint_topic",
                        default="/joint_states", type=str)
    parser.add_argument("--image_topic_1",
                        default="/camera/camera_wrist/color/image_raw", type=str,
                        help="cam0 — wrist camera topic")
    parser.add_argument("--image_topic_2",
                        default="/camera/camera_d405/color/image_raw", type=str,
                        help="cam1 — external camera topic")
    parser.add_argument("--impedance_topic",
                        default="/gello/joint_states", type=str)

    parser.add_argument("--dump_obs", default=None, type=str,
                        help="If set, save per-step diagnostic dump to this .pkl file")

    parser.add_argument("--open_width",            default=0.08,   type=float)
    parser.add_argument("--close_width",           default=0.005,  type=float)
    parser.add_argument("--gripper_speed",         default=0.1,  type=float)
    parser.add_argument("--gripper_force",         default=100.0,  type=float)
    parser.add_argument("--gripper_epsilon_inner", default=0.01,  type=float)
    parser.add_argument("--gripper_epsilon_outer", default=0.01,  type=float)

    args = parser.parse_args()
    args.period = 1.0 / args.hz

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)

    policy = Policy(agent_path, model_name, args)

    env = make_fr3_env_2cam(
        init_node=True,
        hz=args.hz,
        dq_limit=args.dq_limit,
        dg_limit=args.dg_limit,
        settle_time=args.settle_time,
        joint_topic=args.joint_topic,
        image_topic_1=args.image_topic_1,
        image_topic_2=args.image_topic_2,
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
        dump_steps = []

        for t in range(args.T):
            pred_norm = policy._infer(obs.observation)[0]  # (8,) normalized
            action = policy.forward(obs.observation, pred_norm=pred_norm)

            pred_gripper_denorm = pred_norm[7] * policy.scale[7] + policy.loc[7]
            measured_gripper    = float(env.node.get_gripper())

            print(
                f"[STEP {t:04d}] target={np.round(action[:7], 4)}  "
                f"gripper={action[7]:.4f}  "
                f"pred_norm[7]={pred_norm[7]:.4f}  "
                f"measured={measured_gripper:.4f}"
            )

            if args.dump_obs is not None:
                # Encode cam images as JPEG bytes for compact storage
                imgs = obs.observation["images"]
                cam0_jpg = cv2.imencode(".jpg", imgs["cam0"][:, :, ::-1])[1].tobytes()
                cam1_jpg = cv2.imencode(".jpg", imgs["cam1"][:, :, ::-1])[1].tobytes()
                dump_steps.append({
                    "step":               t,
                    "raw_state":          obs.observation["qpos"].copy(),
                    "pred_action_norm":   pred_norm.copy(),
                    "pred_gripper_denorm": float(pred_gripper_denorm),
                    "measured_gripper":   measured_gripper,
                    "cam0_jpg":           cam0_jpg,
                    "cam1_jpg":           cam1_jpg,
                })

            obs = env.step(action)

        if args.dump_obs is not None:
            dump_path = Path(args.dump_obs)
            with open(dump_path, "wb") as f:
                pickle.dump(dump_steps, f)
            print(f"[INFO] Saved {len(dump_steps)}-step dump to {dump_path}")

        env.reset_gripper()

    env.shutdown()


if __name__ == "__main__":
    main()
