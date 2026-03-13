#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Pose2D, PoseStamped
from std_msgs.msg import String, Bool


class BridgeNode(Node):
    """
    bridge_node:
    - Subscribe top-level x, y, yaw command (Pose2D)
    - Convert to PoseStamped in map frame
    - Publish to waypoint_client
    - Forward waypoint_client status / feedback / result back to top-level
    """

    def __init__(self):
        super().__init__('bridge_node')

        # =========================================================
        # Parameters
        # =========================================================
        self.declare_parameter('input_goal_topic', '/top/goal_pose2d')
        self.declare_parameter('output_goal_topic', '/bridge/nav2_goal')
        self.declare_parameter('cancel_in_topic', '/top/nav_cancel')
        self.declare_parameter('cancel_out_topic', '/bridge/nav_cancel')

        self.declare_parameter('client_status_topic', '/bridge/nav_status')
        self.declare_parameter('client_feedback_topic', '/bridge/nav_feedback')
        self.declare_parameter('client_result_topic', '/bridge/nav_result')

        self.declare_parameter('top_status_topic', '/top/nav_status')
        self.declare_parameter('top_feedback_topic', '/top/nav_feedback')
        self.declare_parameter('top_result_topic', '/top/nav_result')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('z_value', 0.0)

        self.input_goal_topic = self.get_parameter('input_goal_topic').value
        self.output_goal_topic = self.get_parameter('output_goal_topic').value
        self.cancel_in_topic = self.get_parameter('cancel_in_topic').value
        self.cancel_out_topic = self.get_parameter('cancel_out_topic').value

        self.client_status_topic = self.get_parameter('client_status_topic').value
        self.client_feedback_topic = self.get_parameter('client_feedback_topic').value
        self.client_result_topic = self.get_parameter('client_result_topic').value

        self.top_status_topic = self.get_parameter('top_status_topic').value
        self.top_feedback_topic = self.get_parameter('top_feedback_topic').value
        self.top_result_topic = self.get_parameter('top_result_topic').value

        self.frame_id = self.get_parameter('frame_id').value
        self.z_value = float(self.get_parameter('z_value').value)

        # =========================================================
        # QoS
        # =========================================================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # =========================================================
        # Publishers
        # =========================================================
        self.goal_pub = self.create_publisher(PoseStamped, self.output_goal_topic, qos)
        self.cancel_pub = self.create_publisher(Bool, self.cancel_out_topic, qos)

        self.top_status_pub = self.create_publisher(String, self.top_status_topic, qos)
        self.top_feedback_pub = self.create_publisher(String, self.top_feedback_topic, qos)
        self.top_result_pub = self.create_publisher(String, self.top_result_topic, qos)

        # =========================================================
        # Subscribers
        # =========================================================
        self.goal_sub = self.create_subscription(
            Pose2D,
            self.input_goal_topic,
            self.goal_callback,
            qos
        )

        self.cancel_sub = self.create_subscription(
            Bool,
            self.cancel_in_topic,
            self.cancel_callback,
            qos
        )

        self.client_status_sub = self.create_subscription(
            String,
            self.client_status_topic,
            self.client_status_callback,
            qos
        )

        self.client_feedback_sub = self.create_subscription(
            String,
            self.client_feedback_topic,
            self.client_feedback_callback,
            qos
        )

        self.client_result_sub = self.create_subscription(
            String,
            self.client_result_topic,
            self.client_result_callback,
            qos
        )

        self.get_logger().info('BridgeNode started.')
        self.get_logger().info(f'  input_goal_topic      : {self.input_goal_topic}')
        self.get_logger().info(f'  output_goal_topic     : {self.output_goal_topic}')
        self.get_logger().info(f'  cancel_in_topic       : {self.cancel_in_topic}')
        self.get_logger().info(f'  cancel_out_topic      : {self.cancel_out_topic}')
        self.get_logger().info(f'  client_status_topic   : {self.client_status_topic}')
        self.get_logger().info(f'  client_feedback_topic : {self.client_feedback_topic}')
        self.get_logger().info(f'  client_result_topic   : {self.client_result_topic}')
        self.get_logger().info(f'  top_status_topic      : {self.top_status_topic}')
        self.get_logger().info(f'  top_feedback_topic    : {self.top_feedback_topic}')
        self.get_logger().info(f'  top_result_topic      : {self.top_result_topic}')
        self.get_logger().info(f'  frame_id              : {self.frame_id}')
        self.get_logger().info(f'  z_value               : {self.z_value:.3f}')

    # =========================================================
    # Utility
    # =========================================================
    def yaw_to_quaternion(self, yaw: float):
        """
        Convert yaw (rad) to quaternion for planar navigation.
        """
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qx, qy, qz, qw

    def build_pose_stamped_from_pose2d(self, msg: Pose2D) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id

        pose_msg.pose.position.x = float(msg.x)
        pose_msg.pose.position.y = float(msg.y)
        pose_msg.pose.position.z = self.z_value

        qx, qy, qz, qw = self.yaw_to_quaternion(float(msg.theta))
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        return pose_msg

    # =========================================================
    # Top -> Bridge -> WaypointClient
    # =========================================================
    def goal_callback(self, msg: Pose2D):
        pose_msg = self.build_pose_stamped_from_pose2d(msg)

        self.goal_pub.publish(pose_msg)

        self.get_logger().info(
            '[Goal Forward] '
            f'Pose2D(x={msg.x:.3f}, y={msg.y:.3f}, yaw={msg.theta:.3f}) '
            f'-> PoseStamped(frame={pose_msg.header.frame_id})'
        )

    def cancel_callback(self, msg: Bool):
        self.cancel_pub.publish(msg)

        self.get_logger().info(
            f'[Cancel Forward] data={msg.data}'
        )

    # =========================================================
    # WaypointClient -> Bridge -> Top
    # =========================================================
    def client_status_callback(self, msg: String):
        self.top_status_pub.publish(msg)
        self.get_logger().info(f'[Status Forward] {msg.data}')

    def client_feedback_callback(self, msg: String):
        self.top_feedback_pub.publish(msg)
        self.get_logger().info(f'[Feedback Forward] {msg.data}')

    def client_result_callback(self, msg: String):
        self.top_result_pub.publish(msg)
        self.get_logger().info(f'[Result Forward] {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = BridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down bridge_node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()