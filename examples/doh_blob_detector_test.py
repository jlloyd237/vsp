# -*- coding: utf-8 -*-
"""Test script for OpenCV contour blob detector (including parameter optimization).
"""

import cv2
import numpy as np

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import SklDoHBlobDetector


def main():
    det = SklDoHBlobDetector()

    with CvVideoCamera(source=1, api_name='DSHOW', is_color=False) as inp, CvVideoDisplay() as out:
        out.open()
        while True:
            frame = inp.read()
            keypoints = det.detect(frame)
            pts = np.array([f.point for f in keypoints])
            sizes = np.array([f.size for f in keypoints])
            print("pts.shape = {}".format(pts.shape))
            print("sizes.shape = {}".format(sizes.shape))

            kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], 2 * kp.size) for kp in keypoints]
            frame_with_kpts = cv2.drawKeypoints(frame, kpts, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out.write(frame_with_kpts)


if __name__ == '__main__':
    main()
