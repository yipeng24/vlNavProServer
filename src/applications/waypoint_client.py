import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

DEPTH_SCALE = 0.001

def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class WaypointClient():
    def __init__(self):
        super().__init__("waypoint_client")
        self.odom_topic = "/odom"
        self.odom_frame = "odom"
        self.base_frame = "base_footprint"

        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5
