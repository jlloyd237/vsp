# -*- coding: utf-8 -*-
"""Test script for video streams.
"""

import time

from vsp.video_stream import CvVideoCamera, CvVideoDisplay, \
    CvVideoInputFile, CvVideoOutputFile


def main():
    with CvVideoCamera(source=1, api_name='V4L2', is_color=False) as inp, \
        CvVideoDisplay() as out:

        inp.set_property('PROP_BUFFERSIZE', 1)
        print(f"camera api: {inp.camera_api}")
        print(f"camera bufsize: {inp.get_property('PROP_BUFFERSIZE')}")
        out.open()

        # give autoexposure a chance to adjust
        for i in range(10):
            frame = inp.read()
            out.write(frame)

        # capture individual frames
        for i in range(5):
            input("Press ENTER to capture frame: ")
            # dump first frame (hardware double-buffering)
            inp.read()
            # use second frame
            frame = inp.read()
            out.write(frame)

    with CvVideoCamera(source=1, api_name='V4L2', is_color=False) as inp, \
        CvVideoDisplay() as out1, \
        CvVideoOutputFile("demo.mp4", is_color=False) as out2:

        inp.set_property('PROP_BUFFERSIZE', 1)  # this will limit frame rate for sequences
        print(f"camera api: {inp.camera_api}")
        print(f"camera bufsize: {inp.get_property('PROP_BUFFERSIZE')}")
        out1.open()
        out2.open()
        start = time.time()
        for i in range(300):
            frame = inp.read()
            out1.write(frame)
            out2.write(frame)
        print("time elapsed = {}".format(time.time() - start))

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
