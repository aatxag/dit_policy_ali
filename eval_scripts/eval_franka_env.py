#!/usr/bin/env python3
"""
FR3 eval environment for dit-policy inference — ONE CAMERA.

cam0 → wrist camera (/camera/camera_wrist/color/image_raw)
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState

from franka_msgs.action import Grasp, Move
from franka_msgs.msg import GraspEpsilon
from rclpy.action import ActionClient


FR3_ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]

FR3_Q_LOW = np.array([-2.3093, -1.5133, -2.4937, -3.0500, -2.4800, 0.8521, -2.6895], dtype=np.float32)
FR3_Q_HIGH = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895], dtype=np.float32)


def ros_image_to_bgr(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == "bgr8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        return arr.reshape(msg.height, msg.width, 3).copy()
    if enc == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        return arr.reshape(msg.height, msg.width, 3)[:, :, ::-1].copy()
    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(arr.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


@dataclass
class SimpleObs:
    observation: dict


class FR3EvalNode1Cam(Node):
    def __init__(
        self,
        joint_topic="/joint_states",
        gripper_topic="/franka_gripper/franka_gripper/joint_states",
        image_topic="/camera/camera_wrist/color/image_raw",
        impedance_topic="/gello/joint_states",
    ):
        super().__init__("fr3_eval_env_1cam")

        self.lock = threading.Lock()
        self.latest_q: Optional[np.ndarray] = None
        self.latest_dq: Optional[np.ndarray] = None
        self.latest_g: Optional[float] = None
        self.latest_img0: Optional[np.ndarray] = None
        self._arm_index_map = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(JointState, joint_topic, self._cb_joint, reliable_qos)
        self.create_subscription(JointState, gripper_topic, self._cb_gripper, sensor_qos)
        self.create_subscription(Image, image_topic, self._cb_image0, reliable_qos)

        self.impedance_pub = self.create_publisher(JointState, impedance_topic, 10)

        self.move_client = ActionClient(self, Move, "/franka_gripper/franka_gripper/move")
        self.grasp_client = ActionClient(self, Grasp, "/franka_gripper/franka_gripper/grasp")

        self._last_gripper_cmd_time = 0.0
        self._last_gripper_mode = None

    def _cb_joint(self, msg: JointState):
        with self.lock:
            if not msg.name or len(msg.name) != len(msg.position):
                return
            if self._arm_index_map is None:
                name_to_idx = {n: i for i, n in enumerate(msg.name)}
                if all(j in name_to_idx for j in FR3_ARM_JOINTS):
                    self._arm_index_map = [name_to_idx[j] for j in FR3_ARM_JOINTS]
            if self._arm_index_map is not None:
                self.latest_q = np.array([msg.position[i] for i in self._arm_index_map], dtype=np.float32)
                if len(msg.velocity) == len(msg.position):
                    self.latest_dq = np.array([msg.velocity[i] for i in self._arm_index_map], dtype=np.float32)

    def _cb_gripper(self, msg: JointState):
        with self.lock:
            if len(msg.position) >= 2:
                self.latest_g = float(msg.position[0] + msg.position[1])
            elif len(msg.position) == 1:
                self.latest_g = float(msg.position[0])

    def _cb_image0(self, msg: Image):
        try:
            img = ros_image_to_bgr(msg)
            with self.lock:
                self.latest_img0 = img
        except Exception as e:
            self.get_logger().warn(f"cam0 decode failed: {e}", once=True)

    def wait_until_ready(self, timeout_sec=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            with self.lock:
                ready = (
                    self.latest_q is not None and
                    self.latest_g is not None and
                    self.latest_img0 is not None
                )
            if ready:
                return True
            if time.time() - t0 > 3.0:
                missing = []
                with self.lock:
                    if self.latest_q is None:
                        missing.append("joint_states")
                    if self.latest_g is None:
                        missing.append("gripper")
                    if self.latest_img0 is None:
                        missing.append("cam0 (wrist)")
                self.get_logger().info(f"Waiting for: {', '.join(missing)}...", once=True)
            time.sleep(0.05)
        return False

    def get_q(self):
        with self.lock:
            if self.latest_q is None:
                raise RuntimeError("No joint state yet")
            return self.latest_q.copy()

    def get_dq(self):
        with self.lock:
            return self.latest_dq.copy() if self.latest_dq is not None else np.zeros(7, dtype=np.float32)

    def get_gripper(self):
        with self.lock:
            if self.latest_g is None:
                raise RuntimeError("No gripper state yet")
            return float(self.latest_g)

    def get_cam0(self):
        with self.lock:
            if self.latest_img0 is None:
                raise RuntimeError("No cam0 image yet")
            return self.latest_img0.copy()

    def publish_impedance_target(self, q7: np.ndarray):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = FR3_ARM_JOINTS
        msg.position = [float(x) for x in q7]
        self.impedance_pub.publish(msg)

    def command_gripper_open(self, width, speed):
        if not self.move_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper move server not available")
            return
        goal = Move.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        self.move_client.send_goal_async(goal)

    def command_gripper_close(self, width, speed, force, eps_inner, eps_outer):
        if not self.grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Gripper grasp server not available")
            return
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        goal.epsilon = GraspEpsilon(inner=float(eps_inner), outer=float(eps_outer))
        self.grasp_client.send_goal_async(goal)


class FR3EvalEnv1Cam:
    def __init__(
        self,
        node: FR3EvalNode1Cam,
        hz=10.0,
        dq_limit=0.15,
        dg_limit=0.02,
        settle_time=1.0,
        open_width=0.08,
        close_width=0.01,
        gripper_speed=0.05,
        gripper_force=60.0,
        gripper_epsilon_inner=0.01,
        gripper_epsilon_outer=0.01,
    ):
        self.node = node
        self.hz = hz
        self.period = 1.0 / hz

        self.dq_limit = float(dq_limit)
        self.dg_limit = float(dg_limit)
        self.settle_time = float(settle_time)

        self.open_width = float(open_width)
        self.close_width = float(close_width)
        self.gripper_speed = float(gripper_speed)
        self.gripper_force = float(gripper_force)
        self.gripper_epsilon_inner = float(gripper_epsilon_inner)
        self.gripper_epsilon_outer = float(gripper_epsilon_outer)

        self.close_trigger = 0.07
        self.open_trigger = 0.075
        self.gripper_command_cooldown = 0.25

        self._last_step_time: Optional[float] = None

    def _make_obs(self):
        q = self.node.get_q().astype(np.float32)
        g = float(self.node.get_gripper())
        img0 = self.node.get_cam0()
        return SimpleObs(observation={
            "images": {"cam0": img0},
            "qpos": np.concatenate([q, [g]]).astype(np.float32),
        })

    def reset(self):
        if not self.node.wait_until_ready(timeout_sec=15.0):
            with self.node.lock:
                missing = []
                if self.node.latest_q is None:
                    missing.append("joint_states")
                if self.node.latest_g is None:
                    missing.append("gripper")
                if self.node.latest_img0 is None:
                    missing.append("cam0 (wrist)")
            raise RuntimeError(f"FR3EvalEnv1Cam not ready after 15s. Missing: {', '.join(missing)}")
        self._last_step_time = None
        print("[INFO] Environment ready. Waiting settle time...")
        time.sleep(self.settle_time)
        return self._make_obs()

    def _command_gripper_from_width(self, g_target: float):
        now = time.time()
        if now - self.node._last_gripper_cmd_time < self.gripper_command_cooldown:
            return

        desired_mode = self.node._last_gripper_mode
        if g_target < self.close_trigger:
            desired_mode = "close"
        elif g_target > self.open_trigger:
            desired_mode = "open"

        if desired_mode == "open" and self.node._last_gripper_mode != "open":
            self.node.command_gripper_open(width=self.open_width, speed=self.gripper_speed)
            self.node._last_gripper_cmd_time = now
            self.node._last_gripper_mode = "open"
            print(f"[GRIPPER] OPEN  -> width={self.open_width:.3f}")

        elif desired_mode == "close" and self.node._last_gripper_mode != "close":
            self.node.command_gripper_close(
                width=self.close_width,
                speed=self.gripper_speed,
                force=self.gripper_force,
                eps_inner=self.gripper_epsilon_inner,
                eps_outer=self.gripper_epsilon_outer,
            )
            self.node._last_gripper_cmd_time = now
            self.node._last_gripper_mode = "close"
            print(f"[GRIPPER] CLOSE -> width={self.close_width:.3f}, force={self.gripper_force:.1f}")

    def step(self, action: np.ndarray) -> SimpleObs:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        q_current = self.node.get_q().astype(np.float32)

        q_target = action[:7]
        g_target = float(action[7])

        q_target = np.clip(q_target, FR3_Q_LOW, FR3_Q_HIGH)
        g_target = np.clip(g_target, 0.0, 0.08)

        dq = np.clip(q_target - q_current, -self.dq_limit, self.dq_limit)
        q_cmd = q_current + dq

        self.node.publish_impedance_target(q_cmd)
        self._command_gripper_from_width(g_target)

        now = time.time()
        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            eff_hz = 1.0 / max(elapsed, 1e-6)
            print(f"[HZ] {eff_hz:.1f}")
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
        self._last_step_time = time.time()

        return self._make_obs()

    def reset_gripper(self):
        self.node.command_gripper_open(width=self.open_width, speed=self.gripper_speed)
        print("[INFO] Reset gripper -> open")

    def shutdown(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def _spin_thread(node):
    rclpy.spin(node)


def make_fr3_env_1cam(
    init_node=True,
    hz=10.0,
    dq_limit=0.15,
    dg_limit=0.02,
    settle_time=1.0,
    joint_topic="/joint_states",
    gripper_topic="/franka_gripper/franka_gripper/joint_states",
    image_topic="/camera/camera_wrist/color/image_raw",
    impedance_topic="/gello/joint_states",
    open_width=0.08,
    close_width=0.01,
    gripper_speed=0.05,
    gripper_force=60.0,
    gripper_epsilon_inner=0.01,
    gripper_epsilon_outer=0.01,
):
    if init_node and not rclpy.ok():
        rclpy.init()

    node = FR3EvalNode1Cam(
        joint_topic=joint_topic,
        gripper_topic=gripper_topic,
        image_topic=image_topic,
        impedance_topic=impedance_topic,
    )

    th = threading.Thread(target=_spin_thread, args=(node,), daemon=True)
    th.start()

    return FR3EvalEnv1Cam(
        node=node,
        hz=hz,
        dq_limit=dq_limit,
        dg_limit=dg_limit,
        settle_time=settle_time,
        open_width=open_width,
        close_width=close_width,
        gripper_speed=gripper_speed,
        gripper_force=gripper_force,
        gripper_epsilon_inner=gripper_epsilon_inner,
        gripper_epsilon_outer=gripper_epsilon_outer,
    )