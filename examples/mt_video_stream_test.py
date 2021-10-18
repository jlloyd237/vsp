# -*- coding: utf-8 -*-
"""Test script for video stream (multi-threading).
"""

import logging

import numpy as np

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, CvVideoOutputFile
from vsp.dataflow import DataflowQueueMT, DataflowFunctionMT

logging.basicConfig(level=logging.INFO, format="(%(threadName)-9s) %(message)s",)


def main():
    
    with CvVideoCamera(is_color=False) as camera,              \
        CvVideoDisplay("display1") as display_1,            \
        CvVideoDisplay("display2") as display_2,            \
        CvVideoOutputFile("demo1.mp4", is_color=False) as video_file_1,     \
        CvVideoOutputFile("demo2.mp4", is_color=False) as video_file_2:

        # Build pipeline
        camera_read_ops = DataflowQueueMT(maxsize=1)
        display_ops = DataflowQueueMT(maxsize=1)
        camera_frames_1 = DataflowQueueMT(maxsize=10)
        camera_frames_2 = DataflowQueueMT(maxsize=10)
        camera_frames_3 = DataflowQueueMT(maxsize=10)
        camera_frames_4 = DataflowQueueMT(maxsize=10)    
        camera_frames_5 = DataflowQueueMT()

        queues = [camera_read_ops, display_ops, camera_frames_1, camera_frames_2,
                  camera_frames_3, camera_frames_4, camera_frames_5]

        video_camera = DataflowFunctionMT(func=camera.read,
                                      in_queues=[camera_read_ops],
                                      out_queues=[camera_frames_1,
                                                  camera_frames_2,
                                                  camera_frames_3,
                                                  camera_frames_4,
                                                  camera_frames_5])

        video_display_1 = DataflowFunctionMT(func=display_1.write,
                                         pre_func=display_1.open,
                                         post_func=display_1.close,
                                         in_queues=[camera_frames_1],
                                         out_queues=[])

        video_display_2 = DataflowFunctionMT(func=display_2.write,
                                         pre_func=display_2.open,
                                         post_func=display_2.close,
                                         in_queues=[display_ops, camera_frames_2],
                                         out_queues=[])

        video_writer_1 = DataflowFunctionMT(func=video_file_1.write,
                                        pre_func=video_file_1.open,
                                        post_func=video_file_1.close,
                                        in_queues=[camera_frames_3],
                                        out_queues=[])

        video_writer_2 = DataflowFunctionMT(func=video_file_2.write,
                                        pre_func=video_file_2.open,
                                        post_func=video_file_2.close,                                        
                                        in_queues=[camera_frames_4],
                                        out_queues=[])

        # Pipeline sorted in topological order
        threads = [video_camera, video_display_1, video_display_2,
                   video_writer_1, video_writer_2]

        # Start pipeline                    
        for t in threads:
            t.start()

        # Issue commands            
        for i in range(300):
            camera_read_ops.put(None)
            display_ops.put(None)

        # Flush and close pipeline        
        for t in threads:
            for q in t.in_queues:
                q.close()
            t.join()
        
        # Pick up results            
        print("Frames captured = {}".format(camera_frames_5.qsize()))
        frame = camera_frames_5.get()
        if isinstance(frame, np.ndarray):
            print("Frame shape = {}".format(frame.shape))
        else:
            print("Frame is not a Numpy array")   

    
if __name__ == '__main__':
    main()
