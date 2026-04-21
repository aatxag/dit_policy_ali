#!/usr/bin/env python3
"""
Conversión recorder sync (2 cámaras) → robobuf con ABSOLUTE ACTIONS
Compatible con recorder_node_two_cam.py V3.
Usa las posiciones articulares (q) y el estado de la pinza para acciones y estado.
No usa EE pose ni grasp poses.
"""

import argparse
import json
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Constante globalak

OPEN_WIDTH = 0.08
CLOSE_WIDTH = 0.005


SYNC_TIMESTAMP_KEYS = [
    "cam1_timestamps",
    "arm_timestamps",
]

# Frekuentzia, resampleoa, hau da 10Hztara, muestra bakoitzak 100ms ro
TARGET_HZ = 10.0
TARGET_DT = 1.0 / TARGET_HZ

# Acepta un frame si está a menos del 55% de TARGET_DT
RESAMPLE_TOLERANCE = 0.55


def resize_and_encode(img_path: Path, size=(256, 256)) -> bytes:
    # Irudia irakurri eta biharra badau redimensionau

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
    """
    Recorderrak gordetzen dau 'gripper_cmd' Int8 (+1 open, -1 close).
    MApeatu ancho fisikotara
    """
    if "gripper_cmd" in data.files:
        cmd = data["gripper_cmd"].astype(np.float32)
    else:

        g = data["gripper"].astype(np.float32)
        cmd = (g > np.median(g)).astype(np.float32)

    # cmd > 0 es abrir (+1). Resto (<=0 o -1) es cerrar
    return np.where(cmd > 0, OPEN_WIDTH, CLOSE_WIDTH).astype(np.float32)


def check_timestamps(data, ep_name):
    """V VAlidatzeko timestampak sinkronizatuta daudela"""

    if "cam0_timestamps" not in data.files:
        print(f"[{ep_name}] ERROR: sin cam0_timestamps")
        return False

    ref = data["cam0_timestamps"].astype(np.float64)
    ok = True


    for key in SYNC_TIMESTAMP_KEYS:
        if key not in data.files:
            continue

        ts = data[key].astype(np.float64)
        n = min(len(ref), len(ts))

        diff = np.abs(ts[:n] - ref[:n]) * 1000.0

        print(
            f"[{ep_name}] {key:<18} "
            f"mean={diff.mean():6.1f}ms max={diff.max():6.1f}ms"
        )
        # Si el desfase máximo de alguna señal es exagerado (>200ms), advertimos
        if diff.max() > 200.0:
            print(f"[{ep_name}] ADVERTENCIA: {key} tiene un desfase alto (>200ms)")
            # No lo marcamos como False (ok=False) para dejar que resample_episode
            # corte el segmento si es necesario, dando más flexibilidad.

    return ok


def resample_episode(cam0_ts, q, g, g_cmd, cam0_files, cam1_files):
    """
    Episodio bakoitza hartu eta 10Hztara remuestreo bat egiten du
    Interpolatutako seinaleekin
    lag eta zuloak kentzen
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

    # Interpolación lineal para el brazo (q) y la medición de la pinza (g)
    q_out = np.column_stack([
        np.interp(t_valid, ts, q[:, j]) for j in range(q.shape[1])
    ]).astype(np.float32)
    g_out = np.interp(t_valid, ts, g).astype(np.float32)

    # Asignación del vecino más cercano para comandos e imágenes
    g_cmd_out = g_cmd[nearest_idx]
    cam0_out = [cam0_files[i] for i in nearest_idx]
    cam1_out = [cam1_files[i] for i in nearest_idx]

    # Divide en segmentos si hay un hueco temporal > 1.5 dt (aprox 150ms)
    split_at = np.where(np.diff(t_valid) > 1.5 * TARGET_DT)[0] + 1
    boundaries = [0] + split_at.tolist() + [len(t_valid)]

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        segments.append({
            "q":          q_out[start:end],
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

    # Episodio bakoitza prozesatu
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

        # Seinaleak irakurri: robota, irudiak eta gripperra
        q = data["q"].astype(np.float32)
        g = data["gripper"].astype(np.float32)
        g_cmd = get_gripper_cmd_width(data)
        cam0_ts = data["cam0_timestamps"].astype(np.float64) if "cam0_timestamps" in data.files else None

        cam0_files = sorted(cam0_dir.glob("*.jpg"))
        cam1_files = sorted(cam1_dir.glob("*.jpg"))

        # Bada ez bada moztu motzena den datura
        t = min(len(q), len(g), len(cam0_files), len(cam1_files))
        q, g, g_cmd = q[:t], g[:t], g_cmd[:t]
        cam0_files, cam1_files = cam0_files[:t], cam1_files[:t]

        if t < 2:
            skipped += 1
            continue

        if cam0_ts is not None and len(cam0_ts) >= t:
            segments, n_dropped = resample_episode(cam0_ts[:t], q, g, g_cmd, cam0_files, cam1_files)
            total_dropped_frames += n_dropped
            if not segments:
                skipped += 1
                continue
        else:
            segments = [{"q": q, "g": g, "g_cmd": g_cmd,
                         "cam0_files": cam0_files, "cam1_files": cam1_files}]

        ep_had_valid_segment = False

        # Transizioak sortu, hau da N estadoekin N-1 trantsizio, azkena ekintza modura
        for seg in segments:
            q_s = seg["q"]
            g_s = seg["g"]
            g_cmd_s = seg["g_cmd"]
            cam0_s = seg["cam0_files"]
            cam1_s = seg["cam1_files"]
            T = len(q_s) - 1 # El último frame se usa como objetivo de la acción

            # Estado: Joints (7) + Anchura actual pinza (1)
            state = np.concatenate([q_s, g_s[:, None]], axis=1)
            # Acción: Joints siguientes (7) + Comando de pinza a enviar (1)
            actions = np.concatenate([q_s[1:], g_cmd_s[1:, None]], axis=1)

            # Trayectoria konstruitu
            traj_out = []
            for i in range(T):
                obs = {
                    "state": state[i],
                    "enc_cam_0": resize_and_encode(cam0_s[i], size),
                    "enc_cam_1": resize_and_encode(cam1_s[i], size),
                }
                traj_out.append((obs, actions[i], 0.0)) # Reward = 0.0

            if len(traj_out) == 0:
                continue

            # Akumulatu guztiak bufferrean eta normalizazioarako datuak
            out_buffer.append(traj_out)
            all_states.append(state[:T])
            all_actions.append(actions)
            ep_had_valid_segment = True

        if not ep_had_valid_segment:
            skipped += 1
            continue

    if len(out_buffer) == 0:
        raise RuntimeError(
            "No hay episodios válidos. "
            "Todos fueron filtrados por timestamps o datos faltantes."
        )

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)

    # GARRANTZITSUA: Normalizazioa
    # Normalización [-1, 1] de las acciones
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
