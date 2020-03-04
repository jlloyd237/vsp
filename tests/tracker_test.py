# -*- coding: utf-8 -*-
"""Simple test script for Tracker classes.
"""

import numpy as np
import scipy.spatial.distance as ssd

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoInputFile, \
    CvVideoOutputFile
from vsp.detector import CvBlobDetector, CvContourBlobDetector
from vsp.tracker import NearestNeighbourTracker
from vsp.processor import CameraStreamProcessor, FileStreamProcessor
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView


def main():
    with CameraStreamProcessor(
            camera=CvVideoCamera(is_color=False),
            pipeline=[
                CvBlobDetector(
                    min_threshold=31.23,
                    max_threshold=207.05,
                    filter_by_color=True,
                    blob_color=255,
                    filter_by_area=True,
                    min_area=17.05,
                    max_area=135.46,
                    filter_by_circularity=True,
                    min_circularity=0.62,
                    filter_by_inertia=True,
                    min_inertia_ratio=0.27,
                    filter_by_convexity=True,
                    min_convexity=0.60,
                ),
                # CvContourBlobDetector(),
                NearestNeighbourTracker(threshold=20),
                KeypointEncoder(),
            ],
            view=KeypointView(color=(0, 255, 0)),
            display=CvVideoDisplay(name='preview'),
            writer=CvVideoOutputFile(is_color=False),
    ) as p:
        # capture sequence of keypoints - ensure sensor has sufficient time to return to
        # its rest position before end of sequence
        keypoints = p.process(num_frames=300, outfile="demo1.mp4")
        #        keypoints = p.process(num_frames=300)
        print(f"keypoints.shape = {keypoints.shape}")

        # check that final keypoint ordering is the same as initial ordering
        init_keypoints, final_keypoints = keypoints[0], keypoints[-1]
        dists = ssd.cdist(final_keypoints, init_keypoints, 'euclidean')
        min_dist_idxs = np.argmin(dists, axis=1)
        print("Test passed: {}".format(np.all(min_dist_idxs == range(len(min_dist_idxs)))))


if __name__ == '__main__':
    main()
