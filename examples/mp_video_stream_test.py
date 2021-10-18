# -*- coding: utf-8 -*-
"""Test script for video stream (multi-processing).
"""

import time
import logging

import numpy as np

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoOutputFile
from vsp.dataflow import DataflowQueueMP, DataflowFunctionMP

logging.basicConfig(level=logging.INFO, format="(%(threadName)-9s) %(message)s",)


def main():
    # Build pipeline
    camera_read_ops = DataflowQueueMP()
    camera_frames_1 = DataflowQueueMP()
    camera_frames_2 = DataflowQueueMP()
    camera_frames_3 = DataflowQueueMP()
    camera_frames_4 = DataflowQueueMP()    
    camera_frames_5 = DataflowQueueMP()

    queues = [camera_read_ops, camera_frames_1, camera_frames_2,
              camera_frames_3, camera_frames_4, camera_frames_5]

    video_camera = DataflowFunctionMP(func=CvVideoCamera.read,
                                   post_func=CvVideoCamera.close,
                                   obj_class=CvVideoCamera,
                                   in_queues=[camera_read_ops],
                                   out_queues=[camera_frames_1,
                                               camera_frames_2,
                                               camera_frames_3,
                                               camera_frames_4,
                                               camera_frames_5])

    video_display_1 = DataflowFunctionMP(func=CvVideoDisplay.write,
                                      pre_func=CvVideoDisplay.open,
                                      post_func=CvVideoDisplay.close,
                                      obj_class=CvVideoDisplay,
                                      obj_kwargs={'name': 'display1'},
                                      in_queues=[camera_frames_1],
                                      out_queues=[])

    video_display_2 = DataflowFunctionMP(func=CvVideoDisplay.write,
                                      pre_func=CvVideoDisplay.open,
                                      post_func=CvVideoDisplay.close,
                                      obj_class=CvVideoDisplay,
                                      obj_kwargs={'name': 'display2'},
                                      in_queues=[camera_frames_2],
                                      out_queues=[])

    video_writer_1 = DataflowFunctionMP(func=CvVideoOutputFile.write,
                                     pre_func=CvVideoOutputFile.open,
                                     post_func=CvVideoOutputFile.close,
                                     obj_class=CvVideoOutputFile,
                                     obj_kwargs={'filename': 'demo1.mp4'},
                                     in_queues=[camera_frames_3],
                                     out_queues=[])

    video_writer_2 = DataflowFunctionMP(func=CvVideoOutputFile.write,
                                     pre_func=CvVideoOutputFile.open,
                                     post_func=CvVideoOutputFile.close,
                                     obj_class=CvVideoOutputFile,
                                     obj_kwargs={'filename': 'demo2.mp4'},
                                     in_queues=[camera_frames_4],
                                     out_queues=[])

    # Pipeline sorted in topological order
    processes = [video_camera, video_display_1, video_display_2,
                 video_writer_1, video_writer_2]

    # Start pipeline                    
    for p in processes:
        p.start()

    # Issue commands            
    for i in range(300):
        camera_read_ops.put(None)

    # Pick up results
    for i in range(300):
        frame = camera_frames_5.get()
        if isinstance(frame, np.ndarray):
            print("Frame shape = {}".format(frame.shape))
        else:
            print("Frame is not a Numpy array")    
    
    # Flush and close pipeline        
    for p in processes:
        for q in p.in_queues:
            q.close()
        p.join()

    
if __name__ == '__main__':
    main()
