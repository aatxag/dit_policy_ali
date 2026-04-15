#!/usr/bin/env python3
"""
Corre el checkpoint offline sobre un dump guardado durante eval online.

Uso:
  python3 replay_dump_offline.py \
      --dump online_dump.pkl \
      --checkpoint bc_finetune/pick_fish_eye_red/wandb_.../pick_fish_eye_red.ckpt

Resultado: para cada step imprime pred_gripper online vs offline y si difieren.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump",       required=True, help="Dump .pkl guardado con --dump_obs")
    parser.add_argument("--checkpoint", required=True, help="Ruta al .ckpt")
    args = parser.parse_args()

    agent_path = Path(args.checkpoint).parent
    model_name = Path(args.checkpoint).name

    with open(agent_path / "agent_config.yaml") as f:
        agent_config = yaml.safe_load(f)
    with open(agent_path / "obs_config.yaml") as f:
        obs_config = yaml.safe_load(f)
    with open(agent_path / "ac_norm.json") as f:
        norm = json.load(f)
    loc   = np.array(norm["loc"],   dtype=np.float32)
    scale = np.array(norm["scale"], dtype=np.float32)

    agent = hydra.utils.instantiate(agent_config)
    sd = torch.load(agent_path / model_name, map_location="cpu", weights_only=False)
    agent.load_state_dict(sd["model"])
    agent = agent.eval().cuda()

    transform = hydra.utils.instantiate(obs_config["transform"])
    img_keys  = obs_config["imgs"]

    with open(args.dump, "rb") as f:
        dump = pickle.load(f)

    print(f"Loaded {len(dump)} steps from {args.dump}")
    print(f"loc[7]={loc[7]:.4f}  scale[7]={scale[7]:.4f}")
    print()
    print(f"{'step':>5}  {'state[7]':>9}  {'online_norm':>11}  {'online_denorm':>13}  "
          f"{'offline_norm':>12}  {'offline_denorm':>14}  {'measured':>9}  match?")
    print("-" * 95)

    for entry in dump:
        t            = entry["step"]
        raw_state    = entry["raw_state"]          # (8,) — what was fed to the model
        online_norm  = entry["pred_action_norm"]   # (8,) — what the online model predicted
        measured_g   = entry["measured_gripper"]

        # Decode images and run offline inference
        imgs = {}
        for i, key in enumerate(img_keys):
            jpg = entry[f"cam{i}_jpg"]
            raw = np.frombuffer(jpg, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            rgb = torch.from_numpy(img[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
            imgs[key] = transform(rgb)[None].cuda()[None]

        state_t = torch.from_numpy(raw_state)[None].float().cuda()
        with torch.no_grad():
            offline_pred = agent.get_actions(imgs, state_t)[0, 0].cpu().numpy()

        on_g_norm  = float(online_norm[7])
        on_g_denorm = on_g_norm * scale[7] + loc[7]
        off_g_norm  = float(offline_pred[7])
        off_g_denorm = off_g_norm * scale[7] + loc[7]
        match = "OK" if abs(on_g_denorm - off_g_denorm) < 0.005 else "DIFF <<<"

        print(f"{t:5d}  {raw_state[7]:9.4f}  {on_g_norm:11.4f}  {on_g_denorm:13.4f}  "
              f"{off_g_norm:12.4f}  {off_g_denorm:14.4f}  {measured_g:9.4f}  {match}")


if __name__ == "__main__":
    main()
