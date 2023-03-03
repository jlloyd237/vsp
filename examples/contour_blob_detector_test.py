# -*- coding: utf-8 -*-
"""Test script for OpenCV contour blob detector (including parameter optimization).
"""

import cv2
import numpy as np

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import CvContourBlobDetector, optimize_contour_blob_detector_params


def main():
    # with CvVideoCamera(source=1, api_name='DSHOW', is_color=False) as inp, CvVideoDisplay() as out: # Windows
    with CvVideoCamera(source=8, frame_size=(640, 480), is_color=False) as inp, CvVideoDisplay() as out:  # Linux
        out.open()
        frames = []
        for i in range(300):
            frame = inp.read()
            out.write(frame)
            frames.append(frame)
        frames = np.array(frames)

    idx = np.random.choice(frames.shape[0], 10, replace=False)
    frames = frames[idx]

    params = optimize_contour_blob_detector_params(frames,
                                                   target_blobs=331,
                                                   blur_kernel_size_half_range=(1, 10),
                                                   thresh_block_size_half_range=(1, 10),
                                                   thresh_constant_range=(-50, 50),
                                                   min_radius_range=(1, 11),
                                                   max_radius_range=(1, 21),
                                                   )
    print(params)
    det = CvContourBlobDetector(**params)

    # with CvVideoCamera(source=1, api_name='DSHOW', is_color=False) as inp, CvVideoDisplay() as out: # Windows
    with CvVideoCamera(source=8, frame_size=(640, 480), is_color=False) as inp, CvVideoDisplay() as out:  # Linux
        out.open()
        for i in range(300):
            frame = inp.read()
            keypoints = det.detect(frame)
            kpts = [cv2.KeyPoint(kp.point[0], kp.point[1], 2 * kp.size) for kp in keypoints]
            frame_with_kpts = cv2.drawKeypoints(frame, kpts, np.array([]), (0, 0, 255),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out.write(frame_with_kpts)


if __name__ == '__main__':
    main()
