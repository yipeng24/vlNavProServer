# viewer.py (你也可以直接写在 ilgp_container.py 里)
import cv2

class RingViewer:
    def __init__(self, ring, buffer_maxlen=30, scale=1.0):
        self.ring = ring
        self.buffer_maxlen = buffer_maxlen
        self.scale = scale
        cv2.namedWindow("ILGP Viewer", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Planer Viewer", cv2.WINDOW_AUTOSIZE)

    
    def tick(self):
          
        packs = self.ring.get_latest(1)
        if not packs:
            print("No image in ring buffer yet.")  
            cv2.waitKey(1)
            return

        p = packs[-1]
        img = p.rgb_bgr

        vis = img.copy()
        cv2.putText(vis, f"stamp(ns): {p.stamp_ns}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(vis, f"ring: {self.ring.size()}/{self.buffer_maxlen}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if abs(self.scale - 1.0) > 1e-3:
            vis = cv2.resize(vis, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("ILGP Viewer", vis)
        cv2.waitKey(1)  # 这个一定要在主线程
