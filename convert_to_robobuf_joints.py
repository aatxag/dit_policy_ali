#!/usr/bin/env python3
"""
Conversión recorder sync (2 cámaras) → robobuf con ABSOLUTE ACTIONS
Compatible con recorder_node_two_cam.py V3.
Usa la Pose del End-Effector (EE) y el estado de la pinza para acciones y estado.
No usa posiciones articulares (joints).
"""

import argparse
import json
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

OPEN_WIDTH = 0.08
CLOSE_WIDTH = 0.005

SYNC_TIMESTAMP_KEYS = [
    "cam1_timestamps",
    "arm_timestamps", # El EE pose se graba al mismo tiempo que el brazo
]

TARGET_HZ = 10.0
TARGET_DT = 1.0 / TARGET_HZ
RESAMPLE_TOLERANCE = 0.55


def resize_and_encode(img_path: Path, size=(256, 256)) -> bytes:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")

    if img.shape[:2] != (size[1], size[0]):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"Error encoding: {img_path}")

    return encoded.tobytes()


def get_gripper_cmd_width(data):
    if "gripper_cmd" in data.files:
        cmd = data["gripper_cmd"].astype(np.float32)
    else:
        g = data["gripper"].astype(np.float32)
        cmd = (g > np.median(g)).astype(np.float32)

    return np.where(cmd > 0, OPEN_WIDTH, CLOSE_WIDTH).astype(np.float32)


def check_timestamps(data, ep_name):
    if "cam0_timestamps" not in data.files:
        print(f"[{ep_name}] ❌ ERROR: sin cam0_timestamps")
        return False

    ref = data["cam0_timestamps"].astype(np.float64)
    ok = True

    for key in SYNC_TIMESTAMP_KEYS:
        if key not in data.files:
            continue

        ts = data[key].astype(np.float64)
        n = min(len(ref), len(ts))
        diff = np.abs(ts[:n] - ref[:n]) * 1000.0

        if diff.max() > 200.0:
            print(f"[{ep_name}] ⚠️ ADVERTENCIA: {key} tiene un desfase alto (>200ms)")

    return ok


def resample_episode(cam0_ts, ee, g, g_cmd, cam0_files, cam1_files):
    """
    Remuestrea a una cuadrícula uniforme de TARGET_HZ.
    Incluye normalización de cuaterniones para la interpolación de la pose EE.
    """
    ts = np.asarray(cam0_ts, dtype=np.float64)
    t0, t_end = ts[0], ts[-1]
    t_grid = np.arange(t0, t_end, TARGET_DT)

    valid_grid_times = []
    nearest_idx = []
    for t in t_grid:
        diffs = np.abs(ts - t)
        idx = int(np.argmin(diffs))
        if diffs[idx] < RESAMPLE_TOLERANCE * TARGET_DT:
            valid_grid_times.append(t)
            nearest_idx.append(idx)

    n_dropped = len(t_grid) - len(valid_grid_times)

    if len(nearest_idx) < 2:
        return [], n_dropped

    t_valid = np.array(valid_grid_times, dtype=np.float64)
    nearest_idx = np.array(nearest_idx, dtype=np.int64)

    # 1. Interpolación lineal para la Pose EE (x, y, z, qx, qy, qz, qw)
    ee_out = np.column_stack([
        np.interp(t_valid, ts, ee[:, j]) for j in range(ee.shape[1])
    ]).astype(np.float32)

    # 1.1 Normalizar los cuaterniones interpolados (índices 3 al 6)
    # Si no normalizamos, el cuaternión deja de ser válido tras la interpolación
    quats = ee_out[:, 3:7]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Evitar división por cero
    ee_out[:, 3:7] = quats / norms

    # 2. Interpolación del medidor de la pinza
    g_out = np.interp(t_valid, ts, g).astype(np.float32)

    # 3. Asignación del vecino más cercano para comandos e imágenes
    g_cmd_out = g_cmd[nearest_idx]
    cam0_out = [cam0_files[i] for i in nearest_idx]
    cam1_out = [cam1_files[i] for i in nearest_idx]

    # Divide en segmentos si hay un hueco temporal
    split_at = np.where(np.diff(t_valid) > 1.5 * TARGET_DT)[0] + 1
    boundaries = [0] + split_at.tolist() + [len(t_valid)]

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        segments.append({
            "ee":         ee_out[start:end],
            "g":          g_out[start:end],
            "g_cmd":      g_cmd_out[start:end],
            "cam0_files": cam0_out[start:end],
            "cam1_files": cam1_out[start:end],
        })

    return segments, n_dropped


def convert_dataset(dataset_dir: str, out_path: str, img_size=256):

    dataset_dir = Path(dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted(episodes_dir.glob("episode_*"))

    out_buffer = []
    all_states = []
    all_actions = []
    skipped = 0
    total_dropped_frames = 0

    size = (img_size, img_size)

    for ep in tqdm(episode_dirs, desc="Episodes"):

        traj = ep / "traj.npz"
        cam0_dir = ep / "cam0_256"
        cam1_dir = ep / "cam1_256"

        if not traj.exists():
            skipped += 1
            continue

        data = np.load(traj, allow_pickle=True)

        if not check_timestamps(data, ep.name):
            skipped += 1
            continue

        # Extraemos EE pose en lugar de joints (q)
        ee = data["ee_pose"].astype(np.float32)
        g = data["gripper"].astype(np.float32)
        g_cmd = get_gripper_cmd_width(data)
        cam0_ts = data["cam0_timestamps"].astype(np.float64) if "cam0_timestamps" in data.files else None

        # Validación rápida por si el robot falló en publicar la pose y se guardó como NaN
        if np.isnan(ee).any():
            print(f"[{ep.name}] ❌ ERROR: ee_pose contiene NaNs. El tópico no estaba disponible.")
            skipped += 1
            continue

        cam0_files = sorted(cam0_dir.glob("*.jpg"))
        cam1_files = sorted(cam1_dir.glob("*.jpg"))

        t = min(len(ee), len(g), len(cam0_files), len(cam1_files))
        ee, g, g_cmd = ee[:t], g[:t], g_cmd[:t]
        cam0_files, cam1_files = cam0_files[:t], cam1_files[:t]

        if t < 2:
            skipped += 1
            continue

        if cam0_ts is not None and len(cam0_ts) >= t:
            segments, n_dropped = resample_episode(cam0_ts[:t], ee, g, g_cmd, cam0_files, cam1_files)
            total_dropped_frames += n_dropped
            if not segments:
                skipped += 1
                continue
        else:
            segments = [{"ee": ee, "g": g, "g_cmd": g_cmd,
                         "cam0_files": cam0_files, "cam1_files": cam1_files}]

        ep_had_valid_segment = False
        for seg in segments:
            ee_s = seg["ee"]
            g_s = seg["g"]
            g_cmd_s = seg["g_cmd"]
            cam0_s = seg["cam0_files"]
            cam1_s = seg["cam1_files"]
            T = len(ee_s) - 1

            # Estado: Pose EE (7) + Anchura actual pinza (1) = 8D
            state = np.concatenate([ee_s, g_s[:, None]], axis=1)
            # Acción: Pose EE siguiente (7) + Comando de pinza a enviar (1) = 8D
            actions = np.concatenate([ee_s[1:], g_cmd_s[1:, None]], axis=1)

            traj_out = []
            for i in range(T):
                obs = {
                    "state": state[i],
                    "enc_cam_0": resize_and_encode(cam0_s[i], size),
                    "enc_cam_1": resize_and_encode(cam1_s[i], size),
                }
                traj_out.append((obs, actions[i], 0.0))

            if len(traj_out) == 0:
                continue

            out_buffer.append(traj_out)
            all_states.append(state[:T])
            all_actions.append(actions)
            ep_had_valid_segment = True

        if not ep_had_valid_segment:
            skipped += 1
            continue

    if len(out_buffer) == 0:
        raise RuntimeError(
            "No hay episodios válidos. Todos fueron filtrados."
        )

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    ac_min = np.percentile(actions, 1, axis=0)
    ac_max = np.percentile(actions, 99, axis=0)

    loc = (ac_min + ac_max) / 2.0
    scale = (ac_max - ac_min).clip(min=1e-6) / 2.0

    for traj in out_buffer:
        for i, (obs, a, r) in enumerate(traj):
            a = np.clip((a - loc) / scale, -1, 1).astype(np.float32)
            traj[i] = (obs, a, r)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "buf.pkl", "wb") as f:
        pkl.dump(out_buffer, f)

    with open(out_path / "ac_norm.json", "w") as f:
        json.dump({"loc": loc.tolist(), "scale": scale.tolist()}, f, indent=2)

    print("\nDONE")
    print("episodes:", len(out_buffer))
    print("frames:", len(actions))
    print("skipped:", skipped)
    print("dropped frames (resampling):", total_dropped_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    convert_dataset(args.dataset_dir, args.out_path, args.img_size)


if __name__ == "__main__":
    main()