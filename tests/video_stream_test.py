# -*- coding: utf-8 -*-
"""Test script for video streams.
"""

import time

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, \
    CvVideoInputFile, CvVideoOutputFile


def main():
    with CvVideoCamera(is_color=False) as inp, \
        CvVideoDisplay() as out1, \
        CvVideoOutputFile("demo.mp4", is_color=False) as out2:

        out1.open()
        out2.open()
        for i in range(300):
            frame = inp.read()
            out1.write(frame)
            out2.write(frame)
            
        time.sleep(5)
    
    with CvVideoInputFile("demo.mp4", is_color=False) as inp, \
        CvVideoDisplay() as out:

        inp.open()
        out.open()
        for frame in inp:
            print(frame.shape)
            out.write(frame)
            
        time.sleep(5)

    
if __name__ == '__main__':
    main()