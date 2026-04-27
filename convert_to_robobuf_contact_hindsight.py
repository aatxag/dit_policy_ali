#!/usr/bin/env python3
"""
Conversión recorder sync (2 cámaras + depth) → robobuf con ABSOLUTE ACTIONS + CONTACT ANCHOR
Compatible con recorder_node_two_cam.py V3.

Usa:
- posiciones articulares (q) y estado de la pinza para estado/acción
- contact anchor 3D extraído desde cam0_depth en el frame de grasp
- transformación del punto cámara → ee → base mediante hand-eye calibration

Esta versión añade:
- --hand_eye.yaml por argumento
- validación de frame_index
- contact_debug.json
- búsqueda robusta del píxel depth válido más cercano al pixel de contacto
"""

import argparse
import json
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ── Constantes ──────────────────────────────────────────────────────────────

OPEN_WIDTH = 0.08
CLOSE_WIDTH = 0.005

SYNC_TIMESTAMP_KEYS = [
    "cam1_timestamps",
    "arm_timestamps",
]

TARGET_HZ = 10.0
TARGET_DT = 1.0 / TARGET_HZ
RESAMPLE_TOLERANCE = 0.55

# Pixel de contacto en imagen 256×256 (u=columna, v=fila)
CONTACT_U = 166
CONTACT_V = 92

# Radio de búsqueda si depth(u,v) == 0
DEPTH_SEARCH_RADIUS = 5

# Valores de depth >= este umbral se consideran inválidos (65535 = no-data en uint16)
DEPTH_MAX_VALID_MM = 5000

# Intrínseca D435i @ 1280×720, reescalada a 256×256
# Fuente: /camera/camera_wrist/color/camera_info  (fx=908.427, fy=908.110, cx=645.173, cy=370.736)
# sx=256/1280=0.2,  sy=256/720≈0.3556
D435I_FX = 181.685
D435I_FY = 322.924
D435I_CX = 129.035
D435I_CY = 131.825


# ── Utilidades geométricas ─────────────────────────────────────────────────

def load_T_cam_to_ee(yaml_path: Path) -> np.ndarray:
    """
    Carga una matriz 4x4 desde YAML.
    Espera una clave:
      T_cam_to_ee_4x4: [[...],[...],[...],[...]]
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe hand-eye YAML: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "T_cam_to_ee_4x4" not in data:
        raise KeyError(
            f"El YAML {yaml_path} no contiene la clave 'T_cam_to_ee_4x4'"
        )

    T = np.array(data["T_cam_to_ee_4x4"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            f"T_cam_to_ee_4x4 debe tener shape (4,4), recibido {T.shape}"
        )

    return T


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convierte pose [x, y, z, qx, qy, qz, qw] a matriz homogénea 4x4."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    if pose7.shape != (7,):
        raise ValueError(f"Pose inválida, shape esperado (7,), recibido {pose7.shape}")

    x, y, z, qx, qy, qz, qw = pose7
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def sanitize_ee_pose(ee_pose_arr: np.ndarray, ep_name: str) -> np.ndarray:
    """
    Reemplaza entradas con cuaternión nulo o inválido por el vecino válido más cercano.
    Devuelve copia saneada. Lanza RuntimeError si no hay ninguna entrada válida.
    """
    arr = ee_pose_arr.copy().astype(np.float64)
    quat = arr[:, 3:]                           # (N, 4) — [qx, qy, qz, qw]
    norms = np.linalg.norm(quat, axis=1)        # (N,)
    valid = norms > 1e-6

    n_bad = int((~valid).sum())
    if n_bad == 0:
        return arr

    if not valid.any():
        raise RuntimeError(f"[{ep_name}] ee_pose sin ninguna entrada válida")

    print(f"  [{ep_name}] ee_pose: {n_bad} entradas con cuaternión nulo → relleno con vecino válido")

    # Para cada índice inválido, usar el vecino válido más cercano
    valid_idx = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
        arr[i] = arr[nearest]

    return arr


def backproject(u: int, v: int, depth_m: float,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Backprojection pinhole. Devuelve punto homogéneo [x,y,z,1] en frame cámara.
    """
    x = (float(u) - float(cx)) * float(depth_m) / float(fx)
    y = (float(v) - float(cy)) * float(depth_m) / float(fy)
    z = float(depth_m)
    return np.array([x, y, z, 1.0], dtype=np.float64)


def find_nearest_valid_depth_pixel(depth_img: np.ndarray, u: int, v: int, radius: int):
    """
    Busca el píxel con depth > 0 más cercano a (u,v) dentro de un radio cuadrado.
    Devuelve (best_u, best_v, best_depth_mm) o None si no encuentra ninguno.

    Criterio:
    - mínima distancia euclídea al centro
    - si empatan, menor profundidad
    """
    H, W = depth_img.shape[:2]

    if not (0 <= u < W and 0 <= v < H):
        return None

    best = None
    best_dist2 = None
    best_depth = None

    u0 = max(0, u - radius)
    u1 = min(W, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(H, v + radius + 1)

    for vv in range(v0, v1):
        for uu in range(u0, u1):
            d = int(depth_img[vv, uu])
            if d <= 0 or d >= DEPTH_MAX_VALID_MM:
                continue

            dist2 = (uu - u) ** 2 + (vv - v) ** 2

            if best is None:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d
                continue

            if dist2 < best_dist2:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d
            elif dist2 == best_dist2 and d < best_depth:
                best = (uu, vv, d)
                best_dist2 = dist2
                best_depth = d

    return best


def get_contact_anchor(
    depth_dir: Path,
    frame_index: int,
    ee_pose_arr: np.ndarray,
    T_cam_to_ee: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    u: int = CONTACT_U,
    v: int = CONTACT_V,
):
    """
    Extrae el contact anchor 3D en frame base.

    Flujo:
    - lee depth del frame grasp
    - usa depth en (u,v), o busca vecino válido más cercano si es 0
    - backprojecta a frame cámara
    - transforma cámara → ee → base
    - devuelve:
        contact_anchor_base: float32 (3,)
        debug_info: dict
    """
    depth_file = depth_dir / f"{frame_index:06d}.png"
    if not depth_file.exists():
        raise FileNotFoundError(f"Depth file no encontrado: {depth_file}")

    depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f"No se pudo leer depth: {depth_file}")

    if depth_img.ndim != 2:
        raise ValueError(
            f"Depth image debe ser monocanal. Shape recibido: {depth_img.shape}"
        )

    H, W = depth_img.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(
            f"Pixel contacto ({u},{v}) fuera del shape depth {(H, W)}"
        )

    used_u, used_v = u, v
    depth_mm = int(depth_img[v, u])
    used_fallback = False

    if depth_mm <= 0 or depth_mm >= DEPTH_MAX_VALID_MM:
        nearest = find_nearest_valid_depth_pixel(depth_img, u, v, DEPTH_SEARCH_RADIUS)
        if nearest is None:
            raise RuntimeError(
                f"Profundidad inválida en ({u},{v}) y no hay vecinos válidos "
                f"en radio {DEPTH_SEARCH_RADIUS}"
            )

        used_u, used_v, depth_mm = nearest
        used_fallback = True

    depth_m = depth_mm / 1000.0

    if frame_index < 0 or frame_index >= len(ee_pose_arr):
        raise IndexError(
            f"frame_index {frame_index} fuera de rango para ee_pose len={len(ee_pose_arr)}"
        )

    p_cam = backproject(used_u, used_v, depth_m, fx, fy, cx, cy)

    # Convención asumida:
    # T_cam_to_ee transforma puntos del frame cámara al frame ee
    # T_ee_to_base transforma puntos del frame ee al frame base

    T_ee_to_base = pose7_to_matrix(ee_pose_arr[frame_index])
    T_cam_to_base = T_ee_to_base @ T_cam_to_ee
    p_ee   = T_cam_to_ee  @ p_cam   # punto en frame EE
    p_base = T_cam_to_base @ p_cam   # punto en frame base

    p_ee_xyz   = p_ee[:3].astype(np.float32)
    p_base_xyz = p_base[:3].astype(np.float32)

    debug_info = {
        "frame_index": int(frame_index),
        "depth_file": str(depth_file),
        "requested_pixel_uv": [int(u), int(v)],
        "used_pixel_uv": [int(used_u), int(used_v)],
        "used_fallback_depth_pixel": bool(used_fallback),
        "depth_mm": int(depth_mm),
        "depth_m": float(depth_m),
        "point_cam_xyz": p_cam[:3].tolist(),
        "point_ee_xyz":  p_ee_xyz.tolist(),
        "point_base_xyz": p_base_xyz.tolist(),
    }

    return p_base_xyz, debug_info


# ── Detección de frame de contacto ─────────────────────────────────────────

def detect_contact_frame(g_trace: np.ndarray, cmd_trace: np.ndarray,
                          min_drop: float = 0.005,
                          start_drop: float = 0.002,
                          slope_eps: float = 0.0005,
                          stable_steps: int = 3) -> int:
    """
    Returns the frame where the gripper stops closing — i.e. physical contact.

    1. Finds cmd_close: first 1→-1 transition in cmd_trace.
    2. Validates the gripper physically drops >= min_drop after cmd_close.
    3. Finds motion_start: first frame where g has dropped >= start_drop from
       g[cmd_close], skipping the flat pre-motion mechanical delay.
    4. Scans forward from motion_start for the first stable window of
       stable_steps frames where |Δg| < slope_eps AND no further significant
       drop (>= min_drop) occurs after the window (filters staircase steps).
    5. Falls back to argmin(g) after cmd_close if no such window is found.
    Returns -1 if no valid close transition or no physical response.
    """
    g = np.asarray(g_trace, dtype=np.float64)
    cmd = np.asarray(cmd_trace)

    open_val = cmd.max()
    close_val = cmd.min()
    if open_val == close_val:
        return -1

    cmd_close = -1
    for i in range(1, len(cmd)):
        if cmd[i - 1] == open_val and cmd[i] == close_val:
            cmd_close = i
            break

    if cmd_close < 0:
        return -1

    if g[cmd_close:].min() > g[cmd_close] - min_drop:
        return -1

    # Skip pre-motion flat plateau (mechanical delay after command)
    g_ref = float(g[cmd_close])
    motion_start = -1
    for i in range(cmd_close + 1, len(g)):
        if g[i] < g_ref - start_drop:
            motion_start = i
            break

    if motion_start < 0:
        return -1

    dg = np.diff(g)
    for i in range(motion_start, len(g) - stable_steps):
        window = dg[i:i + stable_steps]
        if not np.all(np.abs(window) < slope_eps):
            continue
        # Reject staircase steps: require no further significant drop after window
        g_after = g[i + stable_steps:]
        if len(g_after) == 0 or g_after.min() >= g[i] - min_drop:
            return i

    return int(np.argmin(g[cmd_close:]) + cmd_close)


# ── Funciones base del conversor ────────────────────────────────────────────

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
        print(f"[{ep_name}] ERROR: sin cam0_timestamps")
        return False

    ref = data["cam0_timestamps"].astype(np.float64)

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

        if diff.max() > 200.0:
            print(f"[{ep_name}] ADVERTENCIA: {key} tiene un desfase alto (>200ms)")

    return True


def resample_episode(cam0_ts, q, g, g_cmd, cam0_files, cam1_files, ee_pose):

    # Funcion para sincronizar bien, cogemos el primer y el ultimo tiempo
    # y creamos un salto espaciado bueno,luego alineamos los datos mas cercanos
    # en poses interpolamos pero en datos el mas cercano

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

    q_out = np.column_stack([
        np.interp(t_valid, ts, q[:, j]) for j in range(q.shape[1])
    ]).astype(np.float32)

    g_out = np.interp(t_valid, ts, g).astype(np.float32)

    g_cmd_out = g_cmd[nearest_idx]
    cam0_out = [cam0_files[i] for i in nearest_idx]
    cam1_out = [cam1_files[i] for i in nearest_idx]
    ee_pose_out = ee_pose[nearest_idx]  # nearest-neighbor para ee_pose

    split_at = np.where(np.diff(t_valid) > 1.5 * TARGET_DT)[0] + 1
    boundaries = [0] + split_at.tolist() + [len(t_valid)]

    segments = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        segments.append({
            "q": q_out[start:end],
            "g": g_out[start:end],
            "g_cmd": g_cmd_out[start:end],
            "cam0_files": cam0_out[start:end],
            "cam1_files": cam1_out[start:end],
            "ee_pose": ee_pose_out[start:end],
            "orig_idx": nearest_idx[start:end],  # índices originales del episodio
        })

    return segments, n_dropped


# ── Conversión principal ────────────────────────────────────────────────────

def convert_dataset(
    dataset_dir: str,
    out_path: str,
    hand_eye_yaml: str,
    img_size: int = 256,
    fx: float = D435I_FX,
    fy: float = D435I_FY,
    cx: float = D435I_CX,
    cy: float = D435I_CY,
):
    dataset_dir = Path(dataset_dir)
    episodes_dir = dataset_dir / "episodes"
    episode_dirs = sorted(episodes_dir.glob("episode_*"))

    T_cam_to_ee = load_T_cam_to_ee(Path(hand_eye_yaml))
    print(f"[INFO] T_cam_to_ee cargada desde {hand_eye_yaml}")
    print(f"[INFO] Intrinsics @ {img_size}px: fx={fx} fy={fy} cx={cx} cy={cy}")

    out_buffer = []
    all_actions = []
    all_contacts = []

    skipped = 0
    skipped_no_depth = 0
    total_dropped_frames = 0

    size = (img_size, img_size)
    contact_debug = {}

    for ep in tqdm(episode_dirs, desc="Episodes"):
        traj_path = ep / "traj.npz"
        cam0_dir = ep / "cam0_256"
        cam1_dir = ep / "cam1_256"
        cam0_depth_dir = ep / "cam0_depth"

        if not traj_path.exists():
            skipped += 1
            continue

        if not cam0_depth_dir.exists():
            print(f"[{ep.name}] SKIP — sin cam0_depth/")
            skipped += 1
            continue

        data = np.load(traj_path, allow_pickle=True)

        if "ee_pose" not in data.files:
            print(f"[{ep.name}] SKIP — sin ee_pose en traj.npz")
            skipped += 1
            continue

        if not check_timestamps(data, ep.name):
            skipped += 1
            continue

        q = data["q"].astype(np.float32)
        g = data["gripper"].astype(np.float32)
        g_cmd = get_gripper_cmd_width(data)
        cam0_ts = (
            data["cam0_timestamps"].astype(np.float64)
            if "cam0_timestamps" in data.files else None
        )

        cam0_files = sorted(cam0_dir.glob("*.jpg"))
        cam1_files = sorted(cam1_dir.glob("*.jpg"))

        t = min(len(q), len(g), len(cam0_files), len(cam1_files), len(data["ee_pose"]))

        q = q[:t]
        g = g[:t]
        g_cmd = g_cmd[:t]
        cam0_files = cam0_files[:t]
        cam1_files = cam1_files[:t]

        try:
            ee_pose_clean = sanitize_ee_pose(data["ee_pose"][:t], ep.name)
        except RuntimeError as e:
            print(f"[{ep.name}] SKIP — {e}")
            skipped += 1
            continue

        if t < 2:
            print(f"[{ep.name}] SKIP — episodio demasiado corto (t={t})")
            skipped += 1
            continue

        if "gripper_cmd" not in data.files:
            print(f"[{ep.name}] SKIP — sin gripper_cmd en traj.npz")
            skipped += 1
            continue

        frame_index = detect_contact_frame(g, data["gripper_cmd"][:t])
        if frame_index < 0:
            print(f"[{ep.name}] SKIP — no se detecta contacto físico")
            skipped += 1
            continue

        try:
            contact_anchor, debug_info = get_contact_anchor(
                depth_dir=cam0_depth_dir,
                frame_index=frame_index,
                ee_pose_arr=ee_pose_clean,
                T_cam_to_ee=T_cam_to_ee,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                u=CONTACT_U,
                v=CONTACT_V,
            )

        except Exception as e:
            print(f"[{ep.name}] SKIP — error contact anchor: {e}")
            skipped_no_depth += 1
            skipped += 1
            continue

        debug_info["episode_name"] = ep.name
        contact_debug[ep.name] = debug_info

        p_cam_xyz = debug_info["point_cam_xyz"]
        p_ee_xyz  = debug_info["point_ee_xyz"]
        print(
            f"[{ep.name}] contact_anchor "
            f"(cam) = {np.round(p_cam_xyz, 4)} m | "
            f"(EE)  = {np.round(p_ee_xyz,  4)} m | "
            f"(base)= {np.round(contact_anchor, 4).tolist()} m "
            f"(frame {frame_index}, depth={debug_info['depth_mm']}mm)"
        )

        ee_pose_arr = ee_pose_clean

        if cam0_ts is not None and len(cam0_ts) >= t:
            segments, n_dropped = resample_episode(
                cam0_ts[:t], q, g, g_cmd, cam0_files, cam1_files, ee_pose_arr
            )
            total_dropped_frames += n_dropped
            if not segments:
                skipped += 1
                continue
        else:
            segments = [{
                "q": q,
                "g": g,
                "g_cmd": g_cmd,
                "cam0_files": cam0_files,
                "cam1_files": cam1_files,
                "ee_pose": ee_pose_arr,
                "orig_idx": np.arange(t, dtype=np.int64),
            }]

        # Anchor en frame EE del grasp (ya calculado en get_contact_anchor como p_ee_xyz)
        p_ee_at_grasp = np.array(debug_info["point_ee_xyz"], dtype=np.float64)
        p_ee_at_grasp_h = np.append(p_ee_at_grasp, 1.0)
        p_ee_frozen = p_ee_at_grasp.astype(np.float32)

        # T_ee_grasp → base para propagar directamente EE_grasp → EE_i sin pasar por base
        T_ee_grasp_to_base = pose7_to_matrix(ee_pose_arr[frame_index])

        ep_had_valid_segment = False

        for seg in segments:
            q_s      = seg["q"]
            g_s      = seg["g"]
            g_cmd_s  = seg["g_cmd"]
            cam0_s   = seg["cam0_files"]
            cam1_s   = seg["cam1_files"]
            ee_pose_s = seg["ee_pose"]   # (N, 7)

            T_seg = len(q_s) - 1

            state   = np.concatenate([q_s, g_s[:, None]], axis=1)
            actions = np.concatenate([q_s[1:], g_cmd_s[1:, None]], axis=1)

            orig_idx_s = seg["orig_idx"]
            traj_out = []
            for i in range(T_seg):
                if orig_idx_s[i] > frame_index:
                    # Post-grasp: anchor congelado en frame EE del grasp
                    p_ee_i = p_ee_frozen
                else:
                    # Approach + frame de contacto: EE_grasp → base → EE_i
                    T_ee_i_to_base = pose7_to_matrix(ee_pose_s[i])
                    T_base_to_ee_i = np.linalg.inv(T_ee_i_to_base)
                    p_ee_i = (T_base_to_ee_i @ T_ee_grasp_to_base @ p_ee_at_grasp_h)[:3].astype(np.float32)

                obs = {
                    "state": state[i],
                    "enc_cam_0": resize_and_encode(cam0_s[i], size),
                    "enc_cam_1": resize_and_encode(cam1_s[i], size),
                    "contact_anchor": p_ee_i,  # punto en frame EE del step i
                }
                traj_out.append((obs, actions[i], 0.0))

            if len(traj_out) == 0:
                continue

            out_buffer.append(traj_out)
            all_actions.append(actions)
            # Para normalización: acumular todos los valores per-step
            all_contacts.extend([obs["contact_anchor"] for obs, _, _ in traj_out])
            ep_had_valid_segment = True

        if not ep_had_valid_segment:
            skipped += 1
            continue

    if len(out_buffer) == 0:
        raise RuntimeError(
            "No hay episodios válidos. "
            "Todos fueron filtrados por timestamps, datos faltantes o depth inválido."
        )

    actions_all = np.concatenate(all_actions, axis=0)
    # all_contacts contiene un vector (3,) por cada step de todos los trayectos
    contacts_all = np.stack(all_contacts, axis=0)  # (N_steps_total, 3)

    # Normalización acciones [-1, 1]
    ac_min = np.percentile(actions_all, 1, axis=0)
    ac_max = np.percentile(actions_all, 99, axis=0)
    loc = (ac_min + ac_max) / 2.0
    scale = (ac_max - ac_min).clip(min=1e-6) / 2.0

    # Normalización contact anchor: percentil 1-99 → [-1, 1] (igual que acciones)
    c_min = np.percentile(contacts_all, 1, axis=0)
    c_max = np.percentile(contacts_all, 99, axis=0)
    contact_loc   = ((c_min + c_max) / 2.0).astype(np.float32)
    contact_scale = ((c_max - c_min) / 2.0).clip(min=1e-6).astype(np.float32)

    for traj in out_buffer:
        for i, (obs, a, r) in enumerate(traj):
            a = np.clip((a - loc) / scale, -1, 1).astype(np.float32)
            ca_norm = np.clip((obs["contact_anchor"] - contact_loc) / contact_scale, -1, 1).astype(np.float32)
            obs["contact_anchor"] = ca_norm
            traj[i] = (obs, a, r)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "buf.pkl", "wb") as f:
        pkl.dump(out_buffer, f)

    with open(out_path / "ac_norm.json", "w") as f:
        json.dump({"loc": loc.tolist(), "scale": scale.tolist()}, f, indent=2)

    with open(out_path / "contact_norm.json", "w") as f:
        json.dump({
            "loc":   contact_loc.tolist(),
            "scale": contact_scale.tolist(),
        }, f, indent=2)

    with open(out_path / "contact_debug.json", "w") as f:
        json.dump(contact_debug, f, indent=2)

    print("\nDONE")
    print("episodes:", len(out_buffer))
    print("frames:", len(actions_all))
    print("skipped:", skipped)
    print("  of which no valid depth/contact:", skipped_no_depth)
    print("dropped frames (resampling):", total_dropped_frames)
    print("contact_loc:  ", np.round(contact_loc, 4).tolist())
    print("contact_scale:", np.round(contact_scale, 4).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--hand_eye_yaml", default="my_data/hand_eye_result.yaml")

    parser.add_argument("--img_size", type=int, default=256)

    parser.add_argument("--fx", type=float, default=D435I_FX,
                        help="fx escalado a la resolución guardada")
    parser.add_argument("--fy", type=float, default=D435I_FY)
    parser.add_argument("--cx", type=float, default=D435I_CX)
    parser.add_argument("--cy", type=float, default=D435I_CY)

    args = parser.parse_args()

    convert_dataset(
        dataset_dir=args.dataset_dir,
        out_path=args.out_path,
        hand_eye_yaml=args.hand_eye_yaml,
        img_size=args.img_size,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )


if __name__ == "__main__":
    main()