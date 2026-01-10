#!/usr/bin/env python3
# ilgp_container.py

import time
import rclpy
from rclpy.executors import MultiThreadedExecutor

import threading

from image_pool.buffer import ImageRingBuffer
from image_bridge.image_bridge_node import ImageBridgeNode
from xbot_teleop.xbot_teleop_node import XBotTeleop
from .view_img import RingViewer 

def main():
    rclpy.init()

    # 共享 ring：唯一实例
    ring = ImageRingBuffer(
        maxlen=30,
        sync_tolerance_ms=200
    )

    # 把同一个 ring 注入两个节点
    img_node = ImageBridgeNode(ring=ring)
    teleop_node = XBotTeleop(ring=ring)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(img_node)
    executor.add_node(teleop_node)

    # 让 executor 在后台线程跑（ROS 回调随便多线程）
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    viewer = RingViewer(ring, buffer_maxlen=30, window="ILGP Viewer", scale=1.0)
    try:
        while rclpy.ok():
            viewer.tick()
            time.sleep(0.01)  # ~100Hz 刷新（你也可以 0.03 约 30Hz）
    except KeyboardInterrupt:
        pass
    finally:
        img_node.destroy_node()
        teleop_node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
