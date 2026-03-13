#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


class WaypointClient(Node):
    """
    waypoint_client:
    - Subscribe a goal pose from bridge
    - Send goal to Nav2 NavigateToPose action server
    - Publish navigation status / result back to bridge
    """

    def __init__(self):
        super().__init__('waypoint_client')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter('goal_topic', '/bridge/nav2_goal')
        self.declare_parameter('status_topic', '/bridge/nav_status')
        self.declare_parameter('result_topic', '/bridge/nav_result')
        self.declare_parameter('cancel_topic', '/bridge/nav_cancel')
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter('default_frame_id', 'map')

        self.goal_topic = self.get_parameter('goal_topic').get_parameter_value().string_value
        self.status_topic = self.get_parameter('status_topic').get_parameter_value().string_value
        self.result_topic = self.get_parameter('result_topic').get_parameter_value().string_value
        self.cancel_topic = self.get_parameter('cancel_topic').get_parameter_value().string_value
        self.action_name = self.get_parameter('action_name').get_parameter_value().string_value
        self.default_frame_id = self.get_parameter('default_frame_id').get_parameter_value().string_value

        # =========================
        # QoS
        # =========================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # =========================
        # Subscribers / Publishers
        # =========================
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            qos
        )

        self.cancel_sub = self.create_subscription(
            Bool,
            self.cancel_topic,
            self.cancel_callback,
            qos
        )

        self.status_pub = self.create_publisher(String, self.status_topic, qos)
        self.result_pub = self.create_publisher(String, self.result_topic, qos)

        # =========================
        # Action client
        # =========================
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            self.action_name
        )

        self.current_goal_handle = None
        self.current_goal_msg: Optional[PoseStamped] = None
        self.goal_in_progress = False

        self.get_logger().info('WaypointClient started.')
        self.get_logger().info(f'  goal_topic   : {self.goal_topic}')
        self.get_logger().info(f'  cancel_topic : {self.cancel_topic}')
        self.get_logger().info(f'  status_topic : {self.status_topic}')
        self.get_logger().info(f'  result_topic : {self.result_topic}')
        self.get_logger().info(f'  action_name  : {self.action_name}')

    # =========================================================
    # Utility
    # =========================================================
    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'[STATUS] {text}')

    def publish_result(self, text: str):
        msg = String()
        msg.data = text
        self.result_pub.publish(msg)
        self.get_logger().info(f'[RESULT] {text}')

    def pose_to_string(self, pose: PoseStamped) -> str:
        p = pose.pose.position
        q = pose.pose.orientation
        return (
            f'frame={pose.header.frame_id}, '
            f'pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), '
            f'quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})'
        )

    def normalize_quaternion_if_needed(self, pose: PoseStamped):
        q = pose.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)

        # 避免上层传一个全 0 四元数，这玩意儿会把导航系统整得很不高兴
        if norm < 1e-6:
            q.x = 0.0
            q.y = 0.0
            q.z = 0.0
            q.w = 1.0
            return

        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm

    # =========================================================
    # Goal callback
    # =========================================================
    def goal_callback(self, msg: PoseStamped):
        """
        Receive goal pose from bridge and send to Nav2.
        """
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose

        if pose.header.frame_id == '':
            pose.header.frame_id = self.default_frame_id

        self.normalize_quaternion_if_needed(pose)
        self.current_goal_msg = pose

        self.publish_status('received_goal')
        self.get_logger().info(f'Received goal: {self.pose_to_string(pose)}')

        # 如果已有目标在执行，先取消再发新目标
        if self.goal_in_progress and self.current_goal_handle is not None:
            self.publish_status('cancel_current_goal_for_new_goal')
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda future: self._send_goal_after_cancel(future, pose)
            )
        else:
            self.send_goal_to_nav2(pose)

    def _send_goal_after_cancel(self, future, pose: PoseStamped):
        try:
            cancel_response = future.result()
            if cancel_response is not None:
                self.get_logger().info('Previous goal cancel request finished.')
            else:
                self.get_logger().warn('Cancel response is None.')
        except Exception as e:
            self.get_logger().error(f'Error while canceling previous goal: {e}')

        self.goal_in_progress = False
        self.current_goal_handle = None
        self.send_goal_to_nav2(pose)

    # =========================================================
    # Send goal
    # =========================================================
    def send_goal_to_nav2(self, pose: PoseStamped):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=3.0):
            self.publish_status('nav2_action_server_unavailable')
            self.publish_result('failed: nav2 action server unavailable')
            self.get_logger().error(
                f'Action server [{self.action_name}] not available.'
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.publish_status('sending_goal_to_nav2')
        self.get_logger().info('Sending goal to Nav2...')

        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.publish_status('goal_send_exception')
            self.publish_result(f'failed: exception when sending goal: {e}')
            self.get_logger().error(f'Exception in goal_response_callback: {e}')
            return

        if not goal_handle.accepted:
            self.publish_status('goal_rejected')
            self.publish_result('failed: goal rejected by nav2')
            self.get_logger().warn('Goal rejected by Nav2.')
            self.goal_in_progress = False
            self.current_goal_handle = None
            return

        self.current_goal_handle = goal_handle
        self.goal_in_progress = True

        self.publish_status('goal_accepted')
        self.get_logger().info('Goal accepted by Nav2.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    # =========================================================
    # Feedback / Result
    # =========================================================
    def feedback_callback(self, feedback_msg):
        """
        Nav2 feedback callback.
        """
        feedback = feedback_msg.feedback

        # 这里只做简单状态广播，后面你也可以把 distance_remaining / navigation_time 单独发出去
        self.publish_status('navigating')

        try:
            dist = getattr(feedback, 'distance_remaining', None)
            nav_time = getattr(feedback, 'navigation_time', None)

            if dist is not None:
                self.get_logger().info(f'Feedback: distance_remaining={dist:.3f}')
            if nav_time is not None:
                self.get_logger().info(
                    f'Feedback: navigation_time={nav_time.sec}.{nav_time.nanosec:09d}s'
                )
        except Exception as e:
            self.get_logger().warn(f'Failed to parse feedback: {e}')

    def get_result_callback(self, future):
        self.goal_in_progress = False

        try:
            result_wrap = future.result()
            status = result_wrap.status
            result = result_wrap.result
        except Exception as e:
            self.publish_status('result_exception')
            self.publish_result(f'failed: exception when getting result: {e}')
            self.get_logger().error(f'Exception in get_result_callback: {e}')
            self.current_goal_handle = None
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status('goal_reached')
            self.publish_result('success')
            self.get_logger().info('Navigation succeeded.')
        elif status == GoalStatus.STATUS_CANCELED:
            self.publish_status('goal_canceled')
            self.publish_result('canceled')
            self.get_logger().warn('Navigation canceled.')
        elif status == GoalStatus.STATUS_ABORTED:
            self.publish_status('goal_aborted')
            self.publish_result('failed: aborted by nav2')
            self.get_logger().warn('Navigation aborted by Nav2.')
        else:
            self.publish_status(f'goal_finished_with_status_{status}')
            self.publish_result(f'finished_with_status_{status}')
            self.get_logger().warn(f'Navigation finished with status code: {status}')

        self.current_goal_handle = None

    # =========================================================
    # Cancel callback
    # =========================================================
    def cancel_callback(self, msg: Bool):
        if not msg.data:
            return

        if self.current_goal_handle is None or not self.goal_in_progress:
            self.publish_status('no_active_goal_to_cancel')
            self.get_logger().warn('Received cancel request, but no active goal.')
            return

        self.publish_status('cancel_requested')
        self.get_logger().info('Canceling current goal...')

        cancel_future = self.current_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_done_callback)

    def cancel_done_callback(self, future):
        try:
            cancel_response = future.result()
            if cancel_response is not None:
                self.publish_status('cancel_request_sent')
                self.get_logger().info('Cancel request sent successfully.')
            else:
                self.publish_status('cancel_request_failed')
                self.get_logger().warn('Cancel response is None.')
        except Exception as e:
            self.publish_status('cancel_exception')
            self.get_logger().error(f'Exception in cancel_done_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down waypoint_client.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()