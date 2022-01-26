# -*- coding: utf-8 -*-
"""Test script for OpenCV blob detector (including parameter optimization).
"""

import cv2
import numpy as np

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import CvBlobDetector, optimize_blob_detector_params
 

def main():
    with CvVideoCamera(source=1, api_name='DSHOW', is_color=False) as inp, CvVideoDisplay() as out:
        out.open()
        frames = []
        for i in range(300):
            frame = inp.read()
            out.write(frame)
            frames.append(frame)
        frames = np.array(frames)

    idx = np.random.choice(frames.shape[0], 10, replace=False)
    frames = frames[idx]

    params = optimize_blob_detector_params(frames,
                                           target_blobs=331,
                                           min_threshold_range=(0, 300),
                                           max_threshold_range=(0, 300),
                                           min_area_range=(0, 200),
                                           max_area_range=(0, 200),
                                           min_circularity_range=(0.1, 0.9),
                                           min_inertia_ratio_range=(0.1, 0.9),
                                           min_convexity_range=(0.1, 0.9),
                                           )
    print(params)
    det = CvBlobDetector(**params)        

    with CvVideoCamera(source=1, api_name='DSHOW', is_color=False) as inp, CvVideoDisplay() as out:
        for i in range(300):
            frame = inp.read()
            keypoints = det.detect(frame)
            pts = np.array([f.point for f in keypoints])
            sizes = np.array([f.size for f in keypoints])
            print("pts.shape = {}".format(pts.shape))
            print("sizes.shape = {}".format(sizes.shape))
            
            kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]
            frame_with_kpts = cv2.drawKeypoints(frame, kpts, np.array([]), (0,0,255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out.write(frame_with_kpts)


if __name__ == '__main__':
    main()
