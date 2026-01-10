#!/usr/bin/env python3
import math
import numpy as np

from image_pool.image_pool.buffer import ImageRingBuffer
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 如果你要直接让 Nav2 跑：用 action
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose


def yaw_from_quat(q):
    # q: geometry_msgs/Quaternion
    # yaw (z) from quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WaypointPlanner(Node):
    def __init__(self, ring: ImageRingBuffer = None):
        super().__init__("ilgp_waypoint_planner")
        self.ring = ring

        # ========= 配置区（按你习惯写死在脚本内）=========
        self.odom_topic = "/odom"
        self.odom_frame = "odom"
        self.base_frame = "base_footprint"

        # 相机内参（换成真实值）
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5

        # depth 过滤与 waypoint 策略
        self.depth_min_m = 0.25
        self.depth_max_m = 5.0
        self.step_forward_m = None   # 如果你想“总是走一步”，可以设成 0.8；None 表示用像素反投影得到的 x
        self.max_goal_dist_m = 2.0   # 限制一次 waypoint 不要太远（安全）

        # 控制频率
        self.rate_hz = 5.0  # 先 5Hz，之后你可以跟 VLM 的 1Hz 对齐
        # =============================================

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        self.last_odom = None

        # Debug: 发布一个 PoseStamped 给 RViz 看
        self.pub_wp = self.create_publisher(PoseStamped, "/ilgp/waypoint", 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.nav_goal_handle = None
        self.nav_in_flight = False

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        # 状态
        self.state = "IDLE"
        self.pending_instruction = None  # 你输入的指令可以放这里

    def _on_odom(self, msg: Odometry):
        self.last_odom = msg


    def set_instruction(self, text: str):
        # 外部（比如你主线程/teleop）设置新指令
        self.pending_instruction = text

    def _tick(self):
        # 1) 必要条件检查
        if self.last_odom is None:
            return
        if self.pending_instruction is None:
            return
        if self.nav_in_flight:
            return
        if not self.vlm.is_idle():   # 你VLM对象需要提供 is_idle()
            return

        # 2) 触发一次完整链路
        self._run_once(self.pending_instruction)

    def _run_once(self, instruction: str):
        self.state = "INFER"

        packs = self.ring.get_latest(1)
        if not packs:
            self.state = "IDLE"
            return
        p = packs[-1]

        rgb = getattr(p, "rgb_bgr", None)
        depth_m = getattr(p, "depth_m", None)  # 你需要确保 pack 里有 depth_m (meter)
        if rgb is None or depth_m is None:
            self.state = "IDLE"
            return

        # ====== VLM 推理得到 uv ======
        # 你自己实现：输入 instruction + rgb（也可以加4帧历史）
        uv = self.vlm.infer_uv(instruction, rgb)   # -> (u,v)
        if uv is None:
            self.state = "IDLE"
            return
        u, v = int(uv[0]), int(uv[1])

        # ====== depth 解算 waypoint ======
        goal_pose = self._uv_depth_to_goal(u, v, depth_m)
        if goal_pose is None:
            self.state = "IDLE"
            return

        self.pub_goal.publish(goal_pose)

        # ====== 直接发 Nav2 action ======
        self._send_nav2(goal_pose)
        self.state = "NAVIGATING"

    def _run_once(self, instruction: str):
        self.state = "INFER"

        packs = self.ring.get_latest(1)
        if not packs:
            self.state = "IDLE"
            return
        p = packs[-1]

        rgb = getattr(p, "rgb_bgr", None)
        depth_m = getattr(p, "depth_m", None)  # 你需要确保 pack 里有 depth_m (meter)
        if rgb is None or depth_m is None:
            self.state = "IDLE"
            return

        # ====== VLM 推理得到 uv ======
        # 你自己实现：输入 instruction + rgb（也可以加4帧历史）
        uv = self.vlm.infer_uv(instruction, rgb)   # -> (u,v)
        if uv is None:
            self.state = "IDLE"
            return
        u, v = int(uv[0]), int(uv[1])

        # ====== depth 解算 waypoint ======
        goal_pose = self._uv_depth_to_goal(u, v, depth_m)
        if goal_pose is None:
            self.state = "IDLE"
            return

        self.pub_goal.publish(goal_pose)

        # ====== 直接发 Nav2 action ======
        self._send_nav2(goal_pose)
        self.state = "NAVIGATING"

    def _uv_depth_to_goal(self, u: int, v: int, depth_m):
        h, w = depth_m.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        z = float(depth_m[v, u])
        if not (self.depth_min_m <= z <= self.depth_max_m):
            return None

        # 像素反投影（相机坐标）
        x_c = (u - self.cx) * z / self.fx
        z_c = z

        # 简化版：先假设 camera optical 轴≈base_link前向
        # 之后你再用 TF 外参把 (Xc,Yc,Zc) 精确变换到 base_link
        x_bl = z_c
        y_bl = -x_c

        dist = math.hypot(x_bl, y_bl)
        if dist > self.max_goal_dist_m:
            s = self.max_goal_dist_m / dist
            x_bl *= s
            y_bl *= s

        odom_pose = self.last_odom.pose.pose
        x0 = odom_pose.position.x
        y0 = odom_pose.position.y
        yaw0 = yaw_from_quat(odom_pose.orientation)

        gx = x0 + math.cos(yaw0) * x_bl - math.sin(yaw0) * y_bl
        gy = y0 + math.sin(yaw0) * x_bl + math.cos(yaw0) * y_bl

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.goal_frame
        goal.pose.position.x = float(gx)
        goal.pose.position.y = float(gy)
        goal.pose.position.z = 0.0
        goal.pose.orientation = odom_pose.orientation  # 先保持不变
        return goal

    def _send_nav2(self, goal_pose: PoseStamped):
        if not self.nav_client.server_is_ready():
            # action server 还没起来就别发
            self.nav_in_flight = False
            self.state = "IDLE"
            return

        self.nav_in_flight = True
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_pose

        fut = self.nav_client.send_goal_async(nav_goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.nav_in_flight = False
            self.state = "IDLE"
            return
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(self._on_nav_result)

    def _on_nav_result(self, future):
        _ = future.result()
        # 到达/失败都解锁，下一次 VLM idle 会再次触发
        self.nav_in_flight = False
        self.state = "IDLE"