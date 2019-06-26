# -*- coding: utf-8 -*-
"""Test script for DirectShow camera interface.
"""

import time

from vsp.video_stream import CvVideoDisplay, CvVideoInputFile, CvVideoOutputFile
from vsp.dshow_camera import DShowVideoCamera


def main():
    with DShowVideoCamera(format=6, is_color=False) as camera, \
        CvVideoDisplay(name="preview") as display, \
        CvVideoOutputFile(filename="demo.mp4", fps=120, frame_size=(640, 480), is_color=False) as outfile:
            
        print(camera.input_devices)
        print(camera.formats)

        display.open()
        outfile.open()
        start = time.time()
        for i in range(300):
            frame = camera.read()
            display.write(frame)
            outfile.write(frame)
        print("time elapsed = {}".format(time.time() - start))
            
        time.sleep(5)
    
    with CvVideoInputFile(filename="demo.mp4", is_color=False) as inp, \
        CvVideoDisplay(name="preview") as out:

        inp.open()
        out.open()
        for frame in inp:
            out.write(frame)
            
        time.sleep(5)

    
if __name__ == '__main__':
    main()