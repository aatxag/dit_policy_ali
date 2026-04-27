#!/usr/bin/env python3
"""
FR3 eval environment — contact variant.

Extends FR3EvalNode2Cam / FR3EvalEnv2Cam with:
  - Depth image from /camera/camera_wrist/depth/image_rect_raw,
    resized to 256×256 with INTER_NEAREST (identical to the recorder).
  - EE pose from /franka_robot_state_broadcaster/current_pose (PoseStamped).

Public API:
  FR3EvalNode2CamContact  — ROS 2 node
  FR3EvalEnv2CamContact   — gym-like environment wrapper
  make_fr3_env_2cam_contact(...)
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

from eval_franka_env_2cam import FR3EvalEnv2Cam, FR3EvalNode2Cam

_DEPTH_OUT_SIZE = 256


def _pose_to_matrix(pose) -> np.ndarray:
    """geometry_msgs/Pose → 4×4 T_ee_to_base (float64)."""
    p, o = pose.position, pose.orientation
    R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


class FR3EvalNode2CamContact(FR3EvalNode2Cam):
    """Base eval node extended with depth and EE-pose streams."""

    def __init__(
        self,
        joint_topic="/joint_states",
        gripper_topic="/franka_gripper/franka_gripper/joint_states",
        image_topic_1="/camera/camera_wrist/color/image_raw",
        image_topic_2="/camera/camera_ext/color/image_raw",
        impedance_topic="/gello/joint_states",
        depth_topic="/camera/camera_wrist/depth/image_rect_raw",
        ee_pose_topic="/franka_robot_state_broadcaster/current_pose",
    ):
        super().__init__(
            joint_topic=joint_topic,
            gripper_topic=gripper_topic,
            image_topic_1=image_topic_1,
            image_topic_2=image_topic_2,
            impedance_topic=impedance_topic,
        )

        self.latest_depth = None
        self.latest_ee_T = None
        self._bridge = CvBridge()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, depth_topic, self._cb_depth, sensor_qos)
        self.create_subscription(PoseStamped, ee_pose_topic, self._cb_ee_pose, sensor_qos)

    def _cb_depth(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f"depth decode failed: {e}", once=True)

    def _cb_ee_pose(self, msg: PoseStamped):
        T = _pose_to_matrix(msg.pose)
        with self.lock:
            self.latest_ee_T = T

    # ── Public getters ────────────────────────────────────────────────────────

    def get_depth_image_256(self) -> np.ndarray:
        """Depth frame resized to 256×256 with INTER_NEAREST (matches recorder)."""
        with self.lock:
            if self.latest_depth is None:
                raise RuntimeError("No depth image yet")
            d = self.latest_depth.copy()
        return cv2.resize(
            d, (_DEPTH_OUT_SIZE, _DEPTH_OUT_SIZE), interpolation=cv2.INTER_NEAREST
        )

    def get_ee_pose_matrix(self) -> np.ndarray:
        """Current 4×4 T_ee_to_base (EE frame → robot base frame, float64)."""
        with self.lock:
            if self.latest_ee_T is None:
                raise RuntimeError("No EE pose yet")
            return self.latest_ee_T.copy()

    def is_contact_ready(self) -> bool:
        with self.lock:
            return self.latest_depth is not None and self.latest_ee_T is not None


class FR3EvalEnv2CamContact(FR3EvalEnv2Cam):
    """Thin wrapper that exposes depth and EE pose from the contact node."""

    def get_depth_256(self) -> np.ndarray:
        return self.node.get_depth_image_256()

    def get_ee_T(self) -> np.ndarray:
        return self.node.get_ee_pose_matrix()


def make_fr3_env_2cam_contact(
    init_node=True,
    hz=10.0,
    dq_limit=0.15,
    dg_limit=0.02,
    settle_time=0.05,
    joint_topic="/joint_states",
    gripper_topic="/franka_gripper/franka_gripper/joint_states",
    image_topic_1="/camera/camera_wrist/color/image_raw",
    image_topic_2="/camera/camera_ext/color/image_raw",
    impedance_topic="/gello/joint_states",
    depth_topic="/camera/camera_wrist/depth/image_rect_raw",
    ee_pose_topic="/franka_robot_state_broadcaster/current_pose",
    open_width=0.08,
    close_width=0.01,
    gripper_speed=0.05,
    gripper_force=60.0,
    gripper_epsilon_inner=0.01,
    gripper_epsilon_outer=0.01,
) -> FR3EvalEnv2CamContact:
    if init_node and not rclpy.ok():
        rclpy.init()

    node = FR3EvalNode2CamContact(
        joint_topic=joint_topic,
        gripper_topic=gripper_topic,
        image_topic_1=image_topic_1,
        image_topic_2=image_topic_2,
        impedance_topic=impedance_topic,
        depth_topic=depth_topic,
        ee_pose_topic=ee_pose_topic,
    )

    th = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    th.start()

    return FR3EvalEnv2CamContact(
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