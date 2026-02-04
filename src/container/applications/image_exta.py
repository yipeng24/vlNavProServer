import numpy as np

import tf2_ros
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped


DEPTH_SCALE = 0.001
def parse_camera_info(msg: CameraInfo):
    K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    w, h = msg.width, msg.height
    return K, w, h

def deproject_pixel(K, u, v, Z):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return np.array([X, Y, Z], dtype=np.float64)

def tf_to_matrix(T: TransformStamped):
    t = T.transform.translation
    q = T.transform.rotation
    tx, ty, tz = t.x, t.y, t.z
    x, y, z, w = q.x, q.y, q.z, q.w

    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3,:3] = R
    M[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return M

class image_exta:
    def __init__(self):
        super().__init__()
        self.Kc  = self.color_w = self.color_h = None


    def update_camera_info(self, color_info, depth_info, msg: CameraInfo):
        if color_info is not None:
            self.Kc = np.array(color_info.k, dtype=np.float64).reshape(3, 3)
            self.color_h, self.color_w = color_info.height, color_info.width


    def process_uv_to_point(self, uv, depth_img, T_map_from_cam: TransformStamped):
        u, v = uv
        z_raw = depth_img[v, u] * 0.001
        if z_raw <= 0:
            print(f"No depth at pixel {uv}")
            return
        
        Pc = deproject_pixel(self.Kc, u, v, z_raw)

        if T_map_from_cam is None:
            return
        
        M = tf_to_matrix(T_map_from_cam)
        Pc_h = np.array([Pc[0], Pc[1], Pc[2], 1.0], dtype=np.float64)
        Pm = (M @ Pc_h)[:3]
        return Pm
