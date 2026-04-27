#!/usr/bin/env python3
"""
FR3 eval script — contact-conditioned diffusion policy, two cameras.

Workflow per rollout:
  1. User clicks the contact pixel on the live 256×256 wrist image.
  2. Depth at that pixel is read and back-projected to the robot base frame
     using the D435i intrinsics (scaled to 256×256) and hand-eye calibration —
     identical to the training converter.
  3. During the rollout each step:
       pre-grasp  → anchor expressed in current EE frame (vector shrinks toward 0)
       post-grasp → anchor frozen to its value at the moment gripper confirms closed

Usage:
  python eval_franka_2cam_contact.py path/to/checkpoint.ckpt \\
      --hand_eye my_data/hand_eye_result.yaml
"""

import argparse
import json
import os
import pickle
import time
from collections import deque
from pathlib import Path

import cv2
import hydra
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from eval_franka_env_2cam_contact import make_fr3_env_2cam_contact

torch.set_float32_matmul_precision("high")

# ── FR3 joint limits (8-dim: 7 joints + gripper) ─────────────────────────────
FR3_JOINT_LIMITS_LOW = np.array(
    [-2.3093, -1.5133, -2.4937, -3.0500, -2.4800, 0.8521, -2.6895, 0.0],
    dtype=np.float32,
)
FR3_JOINT_LIMITS_HIGH = np.array(
    [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895, 0.08],
    dtype=np.float32,
)

# ── Depth constants (identical to training converter) ─────────────────────────
DEPTH_MAX_VALID_MM = 5000
DEPTH_SEARCH_RADIUS = 5

# ── D435i intrinsics scaled to 256×256 (identical to training) ───────────────
# Source: ros2 topic echo /camera/camera_wrist/color/camera_info @ 1280×720
#   sx = 256/1280 = 0.200,  sy = 256/720 ≈ 0.3556
D435I_FX = 181.685
D435I_FY = 322.924
D435I_CX = 129.035
D435I_CY = 131.825


# ── Geometry helpers ──────────────────────────────────────────────────────────

def load_T_cam_to_ee(yaml_path: str) -> np.ndarray:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    T = np.array(data["T_cam_to_ee_4x4"], dtype=np.float64)
    assert T.shape == (4, 4), f"Expected (4,4), got {T.shape}"
    return T


def find_valid_depth(depth_img: np.ndarray, u: int, v: int, radius: int):
    """Nearest pixel with 0 < depth < DEPTH_MAX_VALID_MM within radius."""
    H, W = depth_img.shape[:2]
    best, best_d2 = None, float("inf")
    for vv in range(max(0, v - radius), min(H, v + radius + 1)):
        for uu in range(max(0, u - radius), min(W, u + radius + 1)):
            d = int(depth_img[vv, uu])
            if d <= 0 or d >= DEPTH_MAX_VALID_MM:
                continue
            d2 = (uu - u) ** 2 + (vv - v) ** 2
            if d2 < best_d2:
                best, best_d2 = (uu, vv, d), d2
    return best


def backproject(u: int, v: int, depth_m: float) -> np.ndarray:
    """Pixel + depth → homogeneous point in camera frame (4,)."""
    return np.array([
        (u - D435I_CX) * depth_m / D435I_FX,
        (v - D435I_CY) * depth_m / D435I_FY,
        depth_m,
        1.0,
    ], dtype=np.float64)


# ── Click UI ─────────────────────────────────────────────────────────────────

def pick_contact_pixel(rgb_256: np.ndarray, depth_256: np.ndarray):
    """
    Shows the wrist RGB image. User clicks the contact pixel.
    The window closes automatically 0.8 s after the click so the script
    continues without requiring a manual window close.
    Returns (u, v) in 256×256 coordinates, or None if closed without clicking.
    """
    rgb_display = rgb_256[:, :, ::-1].copy()  # BGR → RGB
    clicked = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_display)
    ax.set_title("Click the contact point — window closes automatically", fontsize=10)
    ax.axis("off")

    marker = ax.plot([], [], "r+", markersize=18, markeredgewidth=2)[0]
    info_text = ax.text(
        2, 12, "", color="yellow", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6, pad=3),
    )

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        u = max(0, min(255, int(round(event.xdata))))
        v = max(0, min(255, int(round(event.ydata))))

        d_mm = int(depth_256[v, u])
        if 0 < d_mm < DEPTH_MAX_VALID_MM:
            d_str = f"{d_mm} mm"
        else:
            nb = find_valid_depth(depth_256, u, v, DEPTH_SEARCH_RADIUS)
            d_str = f"INVALID → nearest {nb[2]} mm" if nb else "NO VALID DEPTH"

        marker.set_data([u], [v])
        info_text.set_text(f"({u}, {v})  {d_str}")
        fig.canvas.draw_idle()

        clicked.clear()
        clicked.append((u, v))

        # Auto-close after 0.8 s so the user sees their selection
        timer = fig.canvas.new_timer(interval=800)
        timer.single_shot = True
        timer.add_callback(lambda: plt.close(fig))
        timer.start()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()

    return clicked[-1] if clicked else None


# ── Contact anchor lifecycle manager ─────────────────────────────────────────

class ContactAnchor:
    """
    Tracks the contact anchor through the grasp sequence.

    Pre-grasp:  anchor expressed in the current EE frame each step
                  p_ee_i = inv(T_ee_to_base_i) @ p_base_h
    Post-grasp: anchor frozen at the EE-frame value at first grasp detection
                  p_ee_frozen  (object moves with the gripper)
    """

    def __init__(
        self,
        p_base_h: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        close_threshold: float = 0.02,
    ):
        self.p_base_h = p_base_h.astype(np.float64)           # (4,)
        self.contact_loc   = contact_loc.astype(np.float32)   # (3,)
        self.contact_scale = contact_scale.astype(np.float32) # (3,)
        self.close_threshold = close_threshold
        self._frozen = False
        self._p_ee_frozen_norm = None  # (3,) float32, set once

    @classmethod
    def from_env(
        cls,
        env,
        T_cam_to_ee: np.ndarray,
        contact_loc: np.ndarray,
        contact_scale: np.ndarray,
        close_threshold: float = 0.02,
    ) -> "ContactAnchor":
        """
        Interactive setup: shows the wrist image, user clicks the contact pixel,
        reads depth, backprojects and transforms to the robot base frame.
        Returns a ContactAnchor ready for rollout.
        """
        print("\n[CONTACT] Waiting for depth and EE pose to be available...")
        t0 = time.time()
        while not env.node.is_contact_ready():
            if time.time() - t0 > 15.0:
                raise RuntimeError("Timeout waiting for depth/EE pose topics")
            time.sleep(0.1)

        rgb_256 = env.node.get_cam0()
        rgb_256 = cv2.resize(rgb_256, (256, 256), interpolation=cv2.INTER_AREA)
        depth_256 = env.get_depth_256()

        print("[CONTACT] Select the contact point in the window that will open...")
        uv = pick_contact_pixel(rgb_256, depth_256)

        if uv is None:
            raise RuntimeError("No contact point selected — window closed without a click")

        u, v = uv
        depth_mm = int(depth_256[v, u])
        used_u, used_v = u, v

        if depth_mm <= 0 or depth_mm >= DEPTH_MAX_VALID_MM:
            nearest = find_valid_depth(depth_256, u, v, DEPTH_SEARCH_RADIUS)
            if nearest is None:
                raise RuntimeError(
                    f"Depth at ({u},{v}) is invalid ({depth_mm} mm) and no valid "
                    f"neighbor found within radius {DEPTH_SEARCH_RADIUS}"
                )
            used_u, used_v, depth_mm = nearest
            print(f"[CONTACT] Fallback depth pixel: ({used_u},{used_v}) = {depth_mm} mm")

        depth_m = depth_mm / 1000.0
        p_cam_h = backproject(used_u, used_v, depth_m)

        # p_cam → p_ee (via hand-eye)
        p_ee_h = T_cam_to_ee @ p_cam_h

        # p_ee → p_base (via current EE pose)
        T_ee_to_base = env.get_ee_T()
        p_base_h = T_ee_to_base @ p_ee_h

        print(f"[CONTACT] Clicked pixel  : ({u}, {v})")
        print(f"[CONTACT] Used depth px  : ({used_u}, {used_v})  depth={depth_mm} mm")
        print(f"[CONTACT] p_cam          : {np.round(p_cam_h[:3], 4)} m")
        print(f"[CONTACT] p_ee           : {np.round(p_ee_h[:3], 4)} m")
        print(f"[CONTACT] p_base         : {np.round(p_base_h[:3], 4)} m")

        return cls(p_base_h, contact_loc, contact_scale, close_threshold)

    def get_tensor(
        self,
        T_ee_to_base: np.ndarray,
        measured_gripper: float,
        prev_pred_gripper: float | None = None,
    ) -> torch.Tensor:
        """
        Returns a (1, 3) CUDA tensor of the normalized contact anchor for this step.

        Freezes as soon as either:
          - prev_pred_gripper < close_threshold  (policy predicted close last step — matches
            training where freeze = annotated contact frame = gripper command frame)
          - measured_gripper  < close_threshold  (physical fallback)
        Using the predicted command makes the freeze timing consistent with training.
        """
        if not self._frozen:
            phys_close = measured_gripper < self.close_threshold
            if phys_close:
                T_base_to_ee = np.linalg.inv(T_ee_to_base)
                p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
                self._p_ee_frozen_norm = np.clip(
                    (p_ee - self.contact_loc) / self.contact_scale, -1, 1
                )
                self._frozen = True
                trigger = "physical"
                print(
                    f"[CONTACT] Anchor frozen ({trigger})  "
                    f"p_ee={np.round(p_ee, 4)}"
                )

        if self._frozen:
            p_ee_norm = self._p_ee_frozen_norm
        else:
            T_base_to_ee = np.linalg.inv(T_ee_to_base)
            p_ee = (T_base_to_ee @ self.p_base_h)[:3].astype(np.float32)
            p_ee_norm = np.clip((p_ee - self.contact_loc) / self.contact_scale, -1, 1)

        return torch.from_numpy(p_ee_norm).float()[None].cuda()  # (1, 3)


# ── Policy ────────────────────────────────────────────────────────────────────

class Policy:
    def __init__(self, agent_path: str, model_name: str, args):
        self.args = args

        with open(Path(agent_path, "agent_config.yaml"), encoding="utf-8") as f:
            agent_config = yaml.safe_load(f)
        with open(Path(agent_path, "obs_config.yaml"), encoding="utf-8") as f:
            obs_config = yaml.safe_load(f)

        ac_norm_path = (
            Path(args.ac_norm_path).expanduser()
            if args.ac_norm_path
            else Path(agent_path, "ac_norm.json")
        )
        with open(ac_norm_path, encoding="utf-8") as f:
            ac_norm = json.load(f)
        self.loc   = np.array(ac_norm["loc"],   dtype=np.float32)
        self.scale = np.array(ac_norm["scale"], dtype=np.float32)

        # Contact normalization stats (required for contact policy)
        contact_norm_path = Path(agent_path, "contact_norm.json")
        if not contact_norm_path.exists():
            contact_norm_path = Path("contact_norm.json")
        if not contact_norm_path.exists():
            raise FileNotFoundError(
                "contact_norm.json not found in checkpoint directory or cwd. "
                "This checkpoint may not be a contact-conditioned model."
            )
        with open(contact_norm_path, encoding="utf-8") as f:
            cn = json.load(f)
        self.contact_loc   = np.array(cn["loc"],   dtype=np.float32)
        self.contact_scale = np.array(cn["scale"], dtype=np.float32)

        # Build model
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
        self.img_chunk    = int(agent_config.get("imgs_per_cam", 1))

        print(f"[INFO] Checkpoint   : {agent_path}/{model_name}")
        print(f"[INFO] Step         : {save_dict.get('global_step', 'unknown')}")
        print(f"[INFO] ac_norm from : {ac_norm_path}")
        print(f"[INFO] loc          : {np.round(self.loc, 4)}")
        print(f"[INFO] scale        : {np.round(self.scale, 4)}")
        print(f"[INFO] contact_loc  : {np.round(self.contact_loc, 4)}")
        print(f"[INFO] contact_scale: {np.round(self.contact_scale, 4)}")
        print(f"[INFO] img_keys     : {self.img_keys}")
        print(f"[INFO] img_chunk    : {self.img_chunk}")

        self.reset()

    def reset(self):
        self.last_ac = None
        self.img_history = {k: deque(maxlen=self.img_chunk) for k in self.img_keys}

    def _proc_images(self, img_dict, size=(256, 256)):
        for k in self.img_keys:
            self.img_history[k].append(img_dict[k].copy())

        torch_imgs = {}
        for i, k in enumerate(self.img_keys):
            hist = list(self.img_history[k])
            while len(hist) < self.img_chunk:
                hist.insert(0, hist[0])

            frames = []
            for frame in hist:
                bgr = cv2.resize(frame[:, :, :3], size, interpolation=cv2.INTER_AREA)
                rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
                frames.append(rgb)

            stacked = torch.stack(frames, dim=0)
            if self.transform is not None:
                stacked = self.transform(stacked)
            torch_imgs[f"cam{i}"] = stacked[None].cuda()

        return torch_imgs

    def _infer(self, obs: dict, contact_point: torch.Tensor) -> np.ndarray:
        """Returns (pred_horizon, ac_dim) normalized actions."""
        imgs  = self._proc_images(obs["images"])
        state = torch.from_numpy(obs["qpos"]).float()[None].cuda()
        with torch.no_grad():
            ac = self.agent(imgs, state, contact_point=contact_point)
            ac = ac[0].detach().cpu().numpy().astype(np.float32)
        return ac[: self.pred_horizon]

    def forward(
        self,
        obs: dict,
        pred_norm: np.ndarray,
        measured_gripper: float,
    ) -> np.ndarray:
        last_ac = self.last_ac if self.last_ac is not None else pred_norm
        self.last_ac = self.args.gamma * pred_norm + (1.0 - self.args.gamma) * last_ac

        target = self.last_ac * self.scale + self.loc
        target = np.clip(target, FR3_JOINT_LIMITS_LOW, FR3_JOINT_LIMITS_HIGH)

        current = obs["qpos"]
        delta   = target - current
        delta[:7] = np.clip(delta[:7], -self.args.dq_limit, self.args.dq_limit)

        # Amplify arm delta once gripper confirms physically closed
        if measured_gripper < self.args.lift_trigger:
            delta[:7] = np.clip(
                delta[:7] * self.args.lift_scale,
                -self.args.dq_limit,
                self.args.dq_limit,
            )

        delta[7:] = np.clip(delta[7:], -self.args.dg_limit, self.args.dg_limit)
        return current + delta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)

    _default_hand_eye = str(Path(__file__).parent.parent / "my_data" / "hand_eye_result.yaml")
    parser.add_argument("--hand_eye", default=_default_hand_eye,
                        help="Path to hand-eye YAML with T_cam_to_ee_4x4")

    parser.add_argument("--T",             default=300,  type=int)
    parser.add_argument("--num_rollouts",  default=1,    type=int)
    parser.add_argument("--pred_horizon",  default=8,    type=int)
    parser.add_argument("--gamma",         default=0.85, type=float)
    parser.add_argument("--action_idx",    default=0,    type=int,
                        help="Which step of the ac_chunk to execute (0=first). "
                             "Try 2-4 if robot stalls post-grasp.")
    parser.add_argument("--lift_scale",    default=1.0,  type=float,
                        help="Multiply arm delta by this when gripper confirms closed.")
    parser.add_argument("--lift_trigger",  default=0.02, type=float,
                        help="Measured gripper width (m) that activates lift_scale "
                             "and freezes the contact anchor.")

    parser.add_argument("--hz",            default=10.0, type=float)
    parser.add_argument("--dq_limit",      default=0.15, type=float)
    parser.add_argument("--dg_limit",      default=0.02, type=float)
    parser.add_argument("--settle_time",   default=0.05, type=float)

    parser.add_argument("--joint_topic",       default="/joint_states")
    parser.add_argument("--image_topic_1",     default="/camera/camera_wrist/color/image_raw",
                        help="cam0 — wrist camera")
    parser.add_argument("--image_topic_2",     default="/camera/camera_ext/color/image_raw",
                        help="cam1 — external camera")
    parser.add_argument("--depth_topic",       default="/camera/camera_wrist/depth/image_rect_raw")
    parser.add_argument("--ee_pose_topic",     default="/franka_robot_state_broadcaster/current_pose")
    parser.add_argument("--impedance_topic",   default="/gello/joint_states")

    parser.add_argument("--open_width",            default=0.08,  type=float)
    parser.add_argument("--close_width",           default=0.005, type=float)
    parser.add_argument("--gripper_speed",         default=0.1,   type=float)
    parser.add_argument("--gripper_force",         default=100.0, type=float)
    parser.add_argument("--gripper_epsilon_inner", default=0.01,  type=float)
    parser.add_argument("--gripper_epsilon_outer", default=0.01,  type=float)

    parser.add_argument("--ac_norm_path", default=None,
                        help="Override path to ac_norm.json.")
    parser.add_argument("--dump_obs",     default=None,
                        help="If set, save per-step observations to this .pkl file.")

    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = os.path.basename(args.checkpoint)

    T_cam_to_ee = load_T_cam_to_ee(args.hand_eye)
    print(f"[INFO] T_cam_to_ee loaded from {args.hand_eye}")

    policy = Policy(agent_path, model_name, args)

    env = make_fr3_env_2cam_contact(
        init_node=True,
        hz=args.hz,
        dq_limit=args.dq_limit,
        dg_limit=args.dg_limit,
        settle_time=args.settle_time,
        joint_topic=args.joint_topic,
        image_topic_1=args.image_topic_1,
        image_topic_2=args.image_topic_2,
        impedance_topic=args.impedance_topic,
        depth_topic=args.depth_topic,
        ee_pose_topic=args.ee_pose_topic,
        open_width=args.open_width,
        close_width=args.close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_epsilon_inner=args.gripper_epsilon_inner,
        gripper_epsilon_outer=args.gripper_epsilon_outer,
    )

    for rollout_num in range(args.num_rollouts):
        # ── Wait for user confirmation ────────────────────────────────────────
        last_input = None
        while last_input != "y":
            if last_input == "r":
                env.reset()
            last_input = input(
                f"\nRollout {rollout_num + 1}/{args.num_rollouts} — continue? "
                "(y / r=reset gripper): "
            ).strip().lower()

        obs = env.reset()

        # ── Contact point selection ───────────────────────────────────────────
        print(f"\n[ROLLOUT {rollout_num}] Setting up contact anchor...")
        contact = ContactAnchor.from_env(
            env,
            T_cam_to_ee=T_cam_to_ee,
            contact_loc=policy.contact_loc,
            contact_scale=policy.contact_scale,
            close_threshold=args.lift_trigger,
        )

        # Re-reset so the arm starts from a clean observation after the GUI
        obs = env.reset()
        policy.reset()
        dump_steps = []

        print(f"[ROLLOUT {rollout_num}] Starting — {args.T} steps")

        prev_pred_g = None  # predicted gripper from previous step (for freeze trigger)

        for t in range(args.T):
            measured_gripper = float(env.node.get_gripper())
            T_ee_to_base     = env.get_ee_T()

            # Contact anchor for this step (pre-grasp dynamic / post-grasp frozen).
            # Pass prev_pred_g so freeze triggers on gripper command, not physical close.
            contact_tensor = contact.get_tensor(T_ee_to_base, measured_gripper)

            # Infer and select action
            preds     = policy._infer(obs.observation, contact_tensor)
            pred_norm = preds[min(args.action_idx, len(preds) - 1)]
            action    = policy.forward(obs.observation, pred_norm, measured_gripper)

            # Logging
            pred_g_denorm = pred_norm[7] * policy.scale[7] + policy.loc[7]
            prev_pred_g   = float(pred_g_denorm)
            current_q = obs.observation["qpos"][:7]
            delta = action[:7] - current_q
            p_ee_norm = contact_tensor[0].cpu().numpy()
            print(
                f"[STEP {t:04d}] "
                f"current={np.round(current_q, 4)}  "
                f"target={np.round(action[:7], 4)}  "
                f"delta={np.round(delta, 4)}  "
                f"|delta|={np.abs(delta).max():.4f}  "
                f"gripper={action[7]:.4f}  "
                f"pred_g={pred_g_denorm:.4f}  "
                f"measured={measured_gripper:.4f}  "
                f"contact_norm={np.round(p_ee_norm, 3)}"
            )

            if args.dump_obs is not None:
                imgs = obs.observation["images"]
                dump_steps.append({
                    "step": t,
                    "raw_state": obs.observation["qpos"].copy(),
                    "pred_action_norm": pred_norm.copy(),
                    "pred_gripper_denorm": float(pred_g_denorm),
                    "measured_gripper": measured_gripper,
                    "contact_norm": p_ee_norm.copy(),
                    "anchor_frozen": contact._frozen,
                    "cam0_jpg": cv2.imencode(".jpg", imgs["cam0"][:, :, ::-1])[1].tobytes(),
                    "cam1_jpg": cv2.imencode(".jpg", imgs["cam1"][:, :, ::-1])[1].tobytes(),
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