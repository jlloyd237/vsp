# -*- coding: utf-8 -*-
"""Video camera input stream based on Linux v4l2.

Requires the following packages and tools:
    python3-v4l2capture
    libv4l-dev
    gcc (used in setup.py for python3-v4l2capture)
"""

import select

import numpy as np
import cv2
import v4l2capture

from vsp.video_stream import VideoInputStream


class V4l2VideoCamera(VideoInputStream):
    """Video camera input stream based on Linux v4l2.
    """
    def __init__(self,
                 device_path="/dev/video0",
                 frame_size=(640, 480),
                 fourcc = 'MP4V',
                 num_buffers=30,
                 is_color=True,
                 ):
        self.device_path = device_path
        self.frame_size = frame_size
        self.fourcc = fourcc
        self.num_buffers = num_buffers
        self.is_color = is_color
        
        self.open()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cap']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.open() 

    def __call__(self):        
        return self.read()

    @property
    def info(self):
        return self._cap.get_info()
    
    def open(self):
        self._cap = v4l2capture.Video_device(self.device_path)
        self.frame_size = self._cap.set_format(*self.frame_size, fourcc=self.fourcc)
        self._cap.create_buffers(self.num_buffers)
        self._cap.queue_all_buffers()  
        self._cap.start()  
            
    def read(self):
        select.select((self._cap,), (), ())
        frame_data = self._cap.read_and_queue()
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if not self.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
   
    def eof(self):
        return False
    
    def close(self):
        self._cap.stop()
        self._cap.close()
        