#!/usr/bin/env python3
"""
Visualiza el contact anchor (en frame EE) a lo largo de un episodio raw.

Para cada frame i del episodio:
  p_ee_i = inv(T_ee_to_base_i) @ p_base

Esto muestra cómo evoluciona el vector "hacia el punto de contacto"
expresado en el frame del gripper a lo largo del tiempo.

Uso:
  python viz_contact_trajectory.py /path/to/episode_0000 --hand_eye my_data/hand_eye_result.yaml
  python viz_contact_trajectory.py /path/to/episode_0005 --u 140 --v 90 --out fig.png
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── Constantes (igual que el conversor) ─────────────────────────────────────
DEPTH_MAX_VALID_MM = 5000
DEPTH_SEARCH_RADIUS = 10

D435I_FX = 181.685
D435I_FY = 322.924
D435I_CX = 129.035
D435I_CY = 131.825


# ── Geometría ────────────────────────────────────────────────────────────────

def load_T_cam_to_ee(yaml_path: Path) -> np.ndarray:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return np.array(data["T_cam_to_ee_4x4"], dtype=np.float64)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = pose7.astype(np.float64)
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def sanitize_ee_pose(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy().astype(np.float64)
    norms = np.linalg.norm(arr[:, 3:], axis=1)
    valid = norms > 1e-6
    if not valid.any():
        raise RuntimeError("Ninguna entrada de ee_pose válida")
    valid_idx = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
        arr[i] = arr[nearest]
    return arr


def find_valid_depth(depth_img, u, v, radius):
    H, W = depth_img.shape
    best, best_d2 = None, float("inf")
    for vv in range(max(0, v - radius), min(H, v + radius + 1)):
        for uu in range(max(0, u - radius), min(W, u + radius + 1)):
            d = int(depth_img[vv, uu])
            if d <= 0 or d >= DEPTH_MAX_VALID_MM:
                continue
            d2 = (uu - u)**2 + (vv - v)**2
            if d2 < best_d2:
                best, best_d2 = (uu, vv, d), d2
    return best


def backproject(u, v, depth_m, fx, fy, cx, cy):
    return np.array([
        (u - cx) * depth_m / fx,
        (v - cy) * depth_m / fy,
        depth_m,
        1.0,
    ], dtype=np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir")
    parser.add_argument("--hand_eye", default="my_data/hand_eye_result.yaml")
    parser.add_argument("--u",  type=int, default=140)
    parser.add_argument("--v",  type=int, default=90)
    parser.add_argument("--fx", type=float, default=D435I_FX)
    parser.add_argument("--fy", type=float, default=D435I_FY)
    parser.add_argument("--cx", type=float, default=D435I_CX)
    parser.add_argument("--cy", type=float, default=D435I_CY)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    T_cam_to_ee = load_T_cam_to_ee(Path(args.hand_eye))

    # ── Cargar datos del episodio ─────────────────────────────────────────────
    data = np.load(ep / "traj.npz", allow_pickle=True)
    ee_pose_raw = sanitize_ee_pose(data["ee_pose"])
    N = len(ee_pose_raw)

    with open(ep / "grasp_poses.json") as f:
        grasp_poses = json.load(f)
    frame_index = int(grasp_poses[0]["frame_index"])
    print(f"Episodio : {ep.name}")
    print(f"Frames   : {N}")
    print(f"frame_index (grasp): {frame_index}")

    # ── Encontrar p_base desde depth en frame_index ───────────────────────────
    depth_file = ep / "cam0_depth" / f"{frame_index:06d}.png"
    depth_img  = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    depth_mm   = int(depth_img[args.v, args.u])
    used_u, used_v = args.u, args.v

    if depth_mm <= 0 or depth_mm >= DEPTH_MAX_VALID_MM:
        nearest = find_valid_depth(depth_img, args.u, args.v, DEPTH_SEARCH_RADIUS)
        if nearest is None:
            raise RuntimeError(f"Depth inválido en ({args.u},{args.v}) y sin vecinos")
        used_u, used_v, depth_mm = nearest
        print(f"Fallback depth pixel: ({used_u},{used_v}) = {depth_mm}mm")

    depth_m = depth_mm / 1000.0
    p_cam = backproject(used_u, used_v, depth_m, args.fx, args.fy, args.cx, args.cy)
    T_ee_to_base_ref = pose7_to_matrix(ee_pose_raw[frame_index])
    p_base = (T_ee_to_base_ref @ T_cam_to_ee @ p_cam)
    print(f"p_cam   = {np.round(p_cam[:3], 4)} m")
    print(f"p_base  = {np.round(p_base[:3], 4)} m")

    # ── Calcular anchor en frame EE para cada frame ───────────────────────────
    # Pre-grasp: p_ee_i = inv(T_ee_to_base_i) @ p_base  (se acerca a 0)
    # Post-grasp (i >= frame_index): congelado en p_ee_at_grasp  (objeto sostenido)
    T_ee_to_base_grasp = pose7_to_matrix(ee_pose_raw[frame_index])
    T_base_to_ee_grasp = np.linalg.inv(T_ee_to_base_grasp)
    p_ee_at_grasp = (T_base_to_ee_grasp @ p_base)[:3].astype(np.float32)

    anchors = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        if i >= frame_index:
            anchors[i] = p_ee_at_grasp
        else:
            T_ee_to_base_i = pose7_to_matrix(ee_pose_raw[i])
            T_base_to_ee_i = np.linalg.inv(T_ee_to_base_i)
            anchors[i] = (T_base_to_ee_i @ p_base)[:3].astype(np.float32)

    dist = np.linalg.norm(anchors, axis=1)
    steps = np.arange(N)

    # ── Terminal ──────────────────────────────────────────────────────────────
    print(f"\n── p_contact en frame EE por frame ──")
    print(f"{'frame':>6}  {'x (m)':>8}  {'y (m)':>8}  {'z (m)':>8}  {'dist cm':>8}")
    for i in np.unique(np.linspace(0, N - 1, min(N, 25), dtype=int)):
        a = anchors[i]
        marker = " ◄ GRASP" if i == frame_index else ""
        print(f"{i:6d}  {a[0]:8.4f}  {a[1]:8.4f}  {a[2]:8.4f}  {dist[i]*100:8.2f}{marker}")

    # ── Figura ────────────────────────────────────────────────────────────────
    n_imgs = 5
    img_steps = np.linspace(0, N - 1, n_imgs, dtype=int)
    cam0_files = sorted((ep / "cam0_256").glob("*.jpg"))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, n_imgs, height_ratios=[2, 2, 1.8], hspace=0.5, wspace=0.3)

    # Panel 1: x, y, z
    ax_xyz = fig.add_subplot(gs[0, :])
    ax_xyz.plot(steps, anchors[:, 0], label="x", color="C0")
    ax_xyz.plot(steps, anchors[:, 1], label="y", color="C1")
    ax_xyz.plot(steps, anchors[:, 2], label="z", color="C2")
    ax_xyz.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_xyz.axvline(frame_index, color="red", linewidth=1.2, linestyle="--", label=f"grasp (frame {frame_index})")
    ax_xyz.set_ylabel("metros")
    ax_xyz.set_title(f"{ep.name}  —  contact anchor en frame EE  [pixel=({args.u},{args.v}), depth={depth_mm}mm]")
    ax_xyz.legend(loc="upper right", fontsize=9)
    ax_xyz.set_xlim(0, N - 1)
    for s in img_steps:
        ax_xyz.axvline(s, color="gray", linewidth=0.7, linestyle=":")

    # Panel 2: distancia
    ax_dist = fig.add_subplot(gs[1, :])
    ax_dist.plot(steps, dist * 100, color="C3", linewidth=1.5)
    ax_dist.axvline(frame_index, color="red", linewidth=1.2, linestyle="--")
    ax_dist.set_ylabel("distancia (cm)")
    ax_dist.set_xlabel("frame")
    ax_dist.set_title("||p_ee|| — decrece durante approach, congelado post-grasp")
    ax_dist.set_xlim(0, N - 1)
    ax_dist.set_ylim(bottom=0)
    for s in img_steps:
        ax_dist.axvline(s, color="gray", linewidth=0.7, linestyle=":")

    # Panel 3: frames RGB
    for col, s in enumerate(img_steps):
        ax = fig.add_subplot(gs[2, col])
        if s < len(cam0_files):
            img = cv2.imread(str(cam0_files[s]), cv2.IMREAD_COLOR)[:, :, ::-1]
            ax.imshow(img)
        a = anchors[s]
        ax.set_title(
            f"frame {s}\nx={a[0]:.3f}\ny={a[1]:.3f}\nz={a[2]:.3f}\nd={dist[s]*100:.1f}cm",
            fontsize=7,
        )
        ax.axis("off")

    out_png = Path(args.out) if args.out else ep / "viz_contact_trajectory.png"
    plt.savefig(str(out_png), dpi=130, bbox_inches="tight")
    print(f"\nGuardado: {out_png}")


if __name__ == "__main__":
    main()
