# -*- coding: utf-8 -*-
"""Test script for asynchronous video stream processor (multi-processing).
"""

import time

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoOutputFile, CvVideoInputFile
from vsp.detector import CvBlobDetector, CvContourBlobDetector
from vsp.tracker import NearestNeighbourTracker
from vsp.processor import CameraStreamProcessorMP, FileStreamProcessorMP, AsyncProcessor
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView


def main():       
    with AsyncProcessor(CameraStreamProcessorMP(
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
        view=KeypointView(color=(0,255,0)),
        display=CvVideoDisplay(name='preview'),
        writer=CvVideoOutputFile(is_color=False),
        )) as p:        
        
        p.async_process(outfile='demo1.mp4')
        print("Getting on with something else ...")
        time.sleep(5)
        p.async_cancel()
        frames = p.async_result()
        print("frames.shape = {}".format(frames.shape))

    with AsyncProcessor(FileStreamProcessorMP(
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
                NearestNeighbourTracker(threshold=20),
                KeypointEncoder(),
            ],
            view=KeypointView(color=(0, 255, 0)),
            display=CvVideoDisplay(name='replay'),
    )) as p:
        p.async_process(num_frames=100, infile="demo1.mp4")
        print("Getting on with something else ...")
        time.sleep(5)
#        p.async_cancel()
        keypoints = p.async_result()
        print(f"keypoints.shape = {keypoints.shape}")

    
if __name__ == '__main__':
    main()
