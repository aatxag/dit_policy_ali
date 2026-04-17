#!/usr/bin/env python3
"""
Conversión recorder sync (1 cámara) → robobuf con ABSOLUTE ACTIONS.

Estructura esperada del dataset:
  dataset_dir/
    episodes/
      episode_0000/
        traj.npz          # arrays: q (T,7), gripper (T,), gripper_cmd (T,),
                          #         record_timestamps, cam0_timestamps,
                          #         arm_timestamps, gripper_timestamps,
                          #         gripper_cmd_timestamps
        cam0_256/         # imágenes cámara wrist (*.jpg, ordenadas)
      episode_0001/
        ...

Salida:
  out_path/buf.pkl
  out_path/ac_norm.json

Convención de observación y acción:
  obs["state"]  = [q1..q7, gripper_measured]     (8D, valores raw)
  action[:7]    = q[t+1]                         (absolute joint positions)
  action[7]     = gripper_cmd_width[t+1]         (0.08 si cmd>0, 0.005 si cmd<=0)

La acción del gripper representa la intención de abrir/cerrar derivada de
gripper_cmd, no la apertura física medida con retraso.
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

# Streams sincronizados frame-a-frame contra record_timestamps.
# Se usan para decidir si descartar un episodio (skip_ts_ms).
SYNC_TIMESTAMP_KEYS = [
    "cam0_timestamps",   # cam0 USB a ~30 Hz → desfase esperado < 70 ms
    "arm_timestamps",    # /joint_states a ~30 Hz → desfase esperado < 50 ms
]

# Streams que se validan informativamente con un umbral propio más permisivo.
# gripper publica a ~10 Hz → desfases de hasta 150 ms son normales y correctos.
GRIPPER_TIMESTAMP_KEY = "gripper_timestamps"
GRIPPER_SKIP_MS = 200.0   # solo salta si el gripper está MÁS de 200 ms desfasado

AUX_TIMESTAMP_KEYS = [
    "gripper_cmd_timestamps",  # señal de evento humano, no se usa para descartar
]


def resize_and_encode(img_path: Path, size=(256, 256)) -> bytes:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    if img.shape[:2] != (size[1], size[0]):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError(f"No se pudo codificar JPG: {img_path}")
    return encoded.tobytes()


def get_gripper_cmd_width(data, open_width=OPEN_WIDTH, close_width=CLOSE_WIDTH):
    if "gripper_cmd" not in data.files:
        raise KeyError(
            f"'gripper_cmd' no encontrado en traj.npz. Claves: {sorted(data.files)}"
        )
    cmd = data["gripper_cmd"].astype(np.float32)
    unique_cmd = np.unique(cmd)
    print(f"    gripper_cmd unique values: {unique_cmd.tolist()}")
    return np.where(cmd > 0, open_width, close_width).astype(np.float32)


def check_timestamps(data, ep_name, ref_key="record_timestamps",
                     warn_ms=50.0, skip_ms=100.0):
    """
    Valida la sincronización temporal de cada stream.

    - SYNC_TIMESTAMP_KEYS (cam0, arm): se descartan si max > skip_ms.
    - gripper_timestamps: umbral propio GRIPPER_SKIP_MS (10 Hz → hasta 150 ms normal).
    - gripper_cmd_timestamps: solo informativo (señal de evento humano).
    """
    files = set(data.files)
    if ref_key not in files:
        print(f"  [{ep_name}] ERROR: '{ref_key}' no encontrado. Episodio en formato viejo, se salta.")
        return False

    ref_ts = data[ref_key].astype(np.float64)
    ok = True

    # ── Streams de alta frecuencia (cam0, arm) ────────────────
    for key in SYNC_TIMESTAMP_KEYS:
        if key not in files:
            continue
        ts = data[key].astype(np.float64)
        t_cmp = min(len(ref_ts), len(ts))
        if t_cmp == 0:
            print(f"  [{ep_name}] ERROR: '{key}' vacío. Saltando episodio.")
            ok = False
            continue
        diff = np.abs(ts[:t_cmp] - ref_ts[:t_cmp]) * 1000.0
        mean_ms, max_ms = float(diff.mean()), float(diff.max())
        flag = " !!!" if max_ms > warn_ms else ""
        print(f"  [{ep_name}] {key:<24} mean={mean_ms:6.1f}ms  max={max_ms:7.1f}ms{flag}")
        if max_ms > skip_ms:
            print(f"  [{ep_name}] ERROR: desfase > {skip_ms:.0f}ms en {key}. Saltando episodio.")
            ok = False

    # ── Gripper (~10 Hz → hasta 150 ms de desfase es normal) ──
    if GRIPPER_TIMESTAMP_KEY in files:
        ts = data[GRIPPER_TIMESTAMP_KEY].astype(np.float64)
        t_cmp = min(len(ref_ts), len(ts))
        if t_cmp > 0:
            diff = np.abs(ts[:t_cmp] - ref_ts[:t_cmp]) * 1000.0
            mean_ms, max_ms = float(diff.mean()), float(diff.max())
            flag = " !!!" if max_ms > GRIPPER_SKIP_MS else ""
            print(
                f"  [{ep_name}] {GRIPPER_TIMESTAMP_KEY:<24} "
                f"mean={mean_ms:6.1f}ms  max={max_ms:7.1f}ms  "
                f"(umbral={GRIPPER_SKIP_MS:.0f}ms, ~10 Hz){flag}"
            )
            if max_ms > GRIPPER_SKIP_MS:
                print(f"  [{ep_name}] ERROR: gripper desfase > {GRIPPER_SKIP_MS:.0f}ms. Saltando episodio.")
                ok = False

    # ── gripper_cmd_timestamps: solo informativo ──────────────
    if "gripper_cmd_timestamps" in files:
        ts = data["gripper_cmd_timestamps"].astype(np.float64)
        t_cmp = min(len(ref_ts), len(ts))
        if t_cmp > 0:
            diff = np.abs(ts[:t_cmp] - ref_ts[:t_cmp]) * 1000.0
            mean_ms, max_ms = float(diff.mean()), float(diff.max())
            print(
                f"  [{ep_name}] {'gripper_cmd_timestamps':<24} "
                f"mean={mean_ms:6.1f}ms  max={max_ms:7.1f}ms  (info only)"
            )

    return ok

def validate_lengths(data, ep_name, t_expected):
    arrays_to_check = ["q", "gripper", "gripper_cmd"] + SYNC_TIMESTAMP_KEYS + AUX_TIMESTAMP_KEYS
    files = set(data.files)
    for key in arrays_to_check:
        if key not in files:
            continue
        length = len(data[key])
        if length < t_expected:
            print(f"  [{ep_name}] WARN: '{key}' tiene {length} frames, esperado >= {t_expected}.")

def convert_dataset(dataset_dir: str, out_path: str, img_size: int = 256,
                    warn_ts_ms: float = 50.0, skip_ts_ms: float = 100.0):
    dataset_dir = Path(dataset_dir).expanduser()
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted([p for p in episodes_dir.glob("episode_*") if p.is_dir()])

    if not episode_dirs:
        print(f"ERROR: No se encontraron episodios en {episodes_dir}")
        return

    size = (img_size, img_size)
    out_buffer = []
    all_states = []
    all_actions = []
    skipped = 0

    for ep_dir in tqdm(episode_dirs, desc="Convirtiendo episodios"):
        traj_path = ep_dir / "traj.npz"
        cam0_dir = ep_dir / "cam0_256"
        ep_name = ep_dir.name

        missing = []
        if not traj_path.exists():
            missing.append("traj.npz")
        if not cam0_dir.exists():
            missing.append("cam0_256/")
        if missing:
            print(f"  Saltando {ep_name}: falta {', '.join(missing)}")
            skipped += 1
            continue

        data = np.load(traj_path, allow_pickle=True)

        if not check_timestamps(data, ep_name, warn_ms=warn_ts_ms, skip_ms=skip_ts_ms):
            skipped += 1
            continue

        try:
            q = data["q"].astype(np.float32)               # (T, 7)
            g_meas = data["gripper"].astype(np.float32)    # (T,)
            g_cmd_width = get_gripper_cmd_width(data)      # (T,)
        except KeyError as e:
            print(f"  Saltando {ep_name}: {e}")
            skipped += 1
            continue

        cam0_files = sorted(cam0_dir.glob("*.jpg"))

        lengths = {
            "q": len(q),
            "gripper": len(g_meas),
            "gripper_cmd": len(g_cmd_width),
            "cam0_files": len(cam0_files),
        }
        if len(set(lengths.values())) > 1:
            print(
                f"  [{ep_name}] WARN: longitudes inconsistentes — "
                + "  ".join(f"{k}={v}" for k, v in lengths.items())
            )

        t = min(lengths.values())
        if t < 2:
            print(f"  Saltando {ep_name}: solo {t} frames válidos")
            skipped += 1
            continue

        validate_lengths(data, ep_name, t)

        state8 = np.concatenate([q[:t], g_meas[:t, None]], axis=1).astype(np.float32)
        actions_abs = np.concatenate(
            [q[1:t], g_cmd_width[1:t, None]], axis=1
        ).astype(np.float32)

        t_eff = t - 1

        out_traj = []
        for i in range(t_eff):
            out_obs = {
                "state": state8[i],
                "enc_cam_0": resize_and_encode(cam0_files[i], size=size),
            }
            out_traj.append((out_obs, actions_abs[i], 0.0))

        if out_traj:
            out_buffer.append(out_traj)
            all_states.append(state8[:t_eff])
            all_actions.append(actions_abs)

    if not out_buffer:
        print("ERROR: No se convirtió ningún episodio.")
        return

    states_all = np.concatenate(all_states, axis=0)
    actions_all = np.concatenate(all_actions, axis=0)

    ac_min = np.percentile(actions_all, 1, axis=0).astype(np.float32)
    ac_max = np.percentile(actions_all, 99, axis=0).astype(np.float32)
    ac_range = (ac_max - ac_min).clip(min=1e-6).astype(np.float32)
    ac_loc = (ac_min + ac_max) / 2.0
    ac_scale = ac_range / 2.0

    for traj in out_buffer:
        for i, (obs, action, reward) in enumerate(traj):
            norm_ac = np.clip((action - ac_loc) / ac_scale, -1.0, 1.0).astype(np.float32)
            traj[i] = (obs, norm_ac, reward)

    out_path = Path(out_path).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    buf_file = out_path / "buf.pkl"
    with open(buf_file, "wb") as f:
        pkl.dump(out_buffer, f)

    norm_file = out_path / "ac_norm.json"
    with open(norm_file, "w") as f:
        json.dump({"loc": ac_loc.tolist(), "scale": ac_scale.tolist()}, f, indent=2)

    state_names = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_measured"]
    action_names = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_cmd_width"]

    print(f"\n{'='*70}")
    print(f"Buffer guardado en:   {buf_file}")
    print(f"ac_norm guardado en:  {norm_file}")
    print(f"Trayectorias:         {len(out_buffer)}  (saltados: {skipped})")
    print(f"Frames totales:       {len(actions_all)}")
    print(f"{'='*70}")

    print("\n--- States: obs['state'] = [q1..q7, gripper_measured] ---")
    print(f"{'Name':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for i, name in enumerate(state_names):
        print(
            f"{name:<20} {states_all[:,i].mean():>10.4f} {states_all[:,i].std():>10.4f} "
            f"{states_all[:,i].min():>10.4f} {states_all[:,i].max():>10.4f}"
        )

    print("\n--- Actions raw: [q[t+1], gripper_cmd_width[t+1]] ---")
    print(f"{'Name':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for i, name in enumerate(action_names):
        print(
            f"{name:<20} {actions_all[:,i].mean():>10.4f} {actions_all[:,i].std():>10.4f} "
            f"{actions_all[:,i].min():>10.4f} {actions_all[:,i].max():>10.4f}"
        )

    actions_norm = np.clip((actions_all - ac_loc) / ac_scale, -1.0, 1.0)
    print("\n--- Actions NORMALIZED [-1, 1] (stored in buffer) ---")
    print(f"{'Name':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    for i, name in enumerate(action_names):
        print(
            f"{name:<20} {actions_norm[:,i].mean():>10.4f} {actions_norm[:,i].std():>10.4f} "
            f"{actions_norm[:,i].min():>10.4f} {actions_norm[:,i].max():>10.4f}"
        )

    max_abs = float(np.abs(actions_norm).max())
    print(f"\n  Max |normalized action|: {max_abs:.3f}  (compatible con clip_sample=True ✓)")
    print(f"\n  ac_norm gripper_cmd_width: loc={ac_loc[7]:.4f}  scale={ac_scale[7]:.4f}")
    print(f"    open  ({OPEN_WIDTH}) → normalized = {(OPEN_WIDTH - ac_loc[7]) / ac_scale[7]:+.3f}")
    print(f"    close ({CLOSE_WIDTH}) → normalized = {(CLOSE_WIDTH - ac_loc[7]) / ac_scale[7]:+.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convierte dataset de 1 cámara a robobuf con acciones absolutas"
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Directorio raíz del dataset (contiene episodes/)")
    parser.add_argument("--out_path", required=True,
                        help="Directorio de salida para buf.pkl y ac_norm.json")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Tamaño de imagen (default: 256)")
    parser.add_argument("--warn_ts_ms", type=float, default=50.0,
                        help="Umbral de warning para timestamps en ms")
    parser.add_argument("--skip_ts_ms", type=float, default=100.0,
                        help="Umbral para saltar episodio por timestamps en ms")
    args = parser.parse_args()

    convert_dataset(
        args.dataset_dir,
        args.out_path,
        img_size=args.img_size,
        warn_ts_ms=args.warn_ts_ms,
        skip_ts_ms=args.skip_ts_ms,
    )


if __name__ == "__main__":
    main()