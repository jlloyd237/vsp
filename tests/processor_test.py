# -*- coding: utf-8 -*-
"""Simple test script for Stream Processor classes.
"""

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
        keypoints = p.process(num_frames=300, outfile="demo1.mp4", start_frame=1)
        #        keypoints = p.process(num_frames=300)
        print(f"keypoints.shape = {keypoints.shape}")
        print(f"keypoints[0] = {keypoints[0]}")
        init_keypoints = keypoints[0]

        keypoints = p.process(num_frames=150, outfile="demo2.mp4")
        #        keypoints = p.process(num_frames=150)
        print(f"keypoints.shape = {keypoints.shape}")
        print(f"keypoints[0] = {keypoints[0]}")

    with FileStreamProcessor(
            reader=CvVideoInputFile(is_color=False),
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
                NearestNeighbourTracker(threshold=20, keypoints=init_keypoints),
                KeypointEncoder(),
            ],
            view=KeypointView(color=(0, 255, 0)),
            display=CvVideoDisplay(name='preview'),
    ) as p:
        keypoints = p.process(num_frames=100, infile="demo1.mp4", start_frame=1)
        print(f"keypoints.shape = {keypoints.shape}")


if __name__ == '__main__':
    main()
