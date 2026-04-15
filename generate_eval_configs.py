#!/usr/bin/env python3
"""
Generates ac_norm.json and obs_config.yaml for a dit-policy checkpoint.

Usage:
  python3 generate_eval_configs.py \
    --buffer ~/dit-policy/my_data/pick_demo_joint8_abs/buf.pkl \
    --checkpoint_dir ~/dit-policy/bc_finetune/fr3_pick_joint8_ABS_run1/wandb_None_fr3_joint8_resnet_gn_2026-03-24_17-51-19/
"""

import argparse
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import yaml
from robobuf import ReplayBuffer as RB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True, help="Path to buf.pkl")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()

    buffer_path = Path(args.buffer).expanduser()
    ckpt_dir = Path(args.checkpoint_dir).expanduser()

    # ── 1. Load buffer and compute action statistics ──────────────────────────
    print(f"Loading buffer: {buffer_path}")
    with open(buffer_path, "rb") as f:
        traj_list = RB.load_traj_list(pkl.load(f))

    all_actions = []
    for t in traj_list:
        all_actions.append(t.action)

    actions = np.array(all_actions, dtype=np.float32)
    print(f"  Transitions: {actions.shape[0]}, action dim: {actions.shape[1]}")

    loc = actions.mean(axis=0).tolist()
    scale = (actions.std(axis=0) + 1e-8).tolist()

    # ── 2. Write ac_norm.json ─────────────────────────────────────────────────
    ac_norm = {
        "loc": loc,
        "scale": scale,
        "type": "gaussian",
        "action_semantics": "joint_absolute_8d",
    }

    ac_norm_path = ckpt_dir / "ac_norm.json"
    with open(ac_norm_path, "w") as f:
        json.dump(ac_norm, f, indent=2)

    print(f"\nac_norm.json saved to: {ac_norm_path}")
    print(f"  loc:   {[f'{v:.4f}' for v in loc]}")
    print(f"  scale: {[f'{v:.4f}' for v in scale]}")

    if np.array(scale).mean() < 0.01:
        print("  [WARN] scale is very small — are you sure these are absolute (not delta) actions?")

    # ── 3. Write obs_config.yaml ──────────────────────────────────────────────
    obs_config = {
        "imgs": ["cam0"],
        "transform": {
            "_target_": "data4robotics.transforms.get_transform_by_name",
            "name": "preproc",
        },
    }

    obs_config_path = ckpt_dir / "obs_config.yaml"
    with open(obs_config_path, "w") as f:
        yaml.dump(obs_config, f, default_flow_style=False)

    print(f"obs_config.yaml saved to: {obs_config_path}")
    print("\nDone — checkpoint directory is ready for eval.")


if __name__ == "__main__":
    main()
