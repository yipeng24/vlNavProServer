import sys
print("PYTHON EXEC:", sys.executable)
print("PYTHON PATH 0:", sys.path[0])
import time
import rclpy
from rclpy.executors import MultiThreadedExecutor

import threading

from image_pool.buffer import ImageRingBuffer

from vlm_service.vlm_genai import VLMClient

from image_bridge.image_bridge_node import ImageBridgeNode
from xbot_teleop.xbot_teleop_node import XBotTeleop
from waypoint_planner.waypoint_planner_node import WaypointPlanner

from .view_img import RingViewer 

def main():
    rclpy.init()

    # 共享 ring：唯一实例
    ring = ImageRingBuffer(
        maxlen=30,
        sync_tolerance_ms=200
    )
    vlm_client = VLMClient()  # 如果你要在 container 里直接用 VLMClient，可以在这里实例化

    # 把同一个 ring 注入两个节点
    img_node = ImageBridgeNode(ring=ring)
    teleop_node = XBotTeleop(ring=ring)
    waypoint_planner_node = WaypointPlanner(ring=ring, vlm=vlm_client)
    executor = MultiThreadedExecutor(num_threads=4)

    executor.add_node(img_node)
    executor.add_node(teleop_node)
    executor.add_node(waypoint_planner_node)

    # 让 executor 在后台线程跑（ROS 回调随便多线程）
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    viewer = RingViewer(ring, buffer_maxlen=30, scale=1.0)
    
    try:
        while rclpy.ok():
            # rclpy.logging.get_logger("ilgp_container").info("Container running...")
            # rclpy.logging.get_logger("ilgp_container").info(f"Ring buffer size: {ring.size()}/30")
            viewer.tick()
            time.sleep(0.1)  # ~100Hz 刷新（你也可以 0.03 约 30Hz）
    except KeyboardInterrupt:
        pass
    finally:
        img_node.destroy_node()
        teleop_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
