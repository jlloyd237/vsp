# -*- coding: utf-8 -*-
"""Test script for v4l2 camera interface.
"""

import time

from vsp.video_stream import CvVideoDisplay, CvVideoInputFile, CvVideoOutputFile
from vsp.v4l2_camera import V4l2VideoCamera


def main():
    with V4l2VideoCamera(device_path="/dev/video1", frame_size=(640, 480),
                         num_buffers=1, is_color=False) as camera, \
        CvVideoDisplay(name="preview") as display:

        display.open()

        # give autoexposure a chance to adjust
        for i in range(10):
            frame = camera.read()
            display.write(frame)

        # capture individual frames
        for i in range(5):
            input("Press ENTER to capture frame: ")
            # dump first frame (hardware double-buffering)
            camera.read()
            # use second frame
            frame = camera.read()
            display.write(frame)

    with V4l2VideoCamera(device_path="/dev/video1", frame_size=(640, 480),
                         num_buffers=1, is_color=False) as camera, \
        CvVideoDisplay(name="preview") as display, \
        CvVideoOutputFile(filename="demo.mp4", fps=120, frame_size=(640, 480),
                          is_color=False) as outfile:

        print(camera.info)
        print(camera.frame_size)
            
        display.open()
        outfile.open()
        start = time.time()
        for i in range(300):
            frame = camera.read()
            display.write(frame)
            outfile.write(frame)
        print("time elapsed = {}".format(time.time() - start))

        time.sleep(5)
    
    with CvVideoInputFile("demo.mp4", is_color=False) as inp, \
        CvVideoDisplay(name="preview") as out:

        inp.open()
        out.open()
        for frame in inp:
            out.write(frame)
            
        time.sleep(5)

    
if __name__ == '__main__':
    main()
