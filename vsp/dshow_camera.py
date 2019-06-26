# -*- coding: utf-8 -*-
"""Video camera input stream based on Microsoft DirectShow.
"""

import threading

import cv2

from vsp.video_stream import VideoInputStream
from vsp.pygrabber.dshow_graph import FilterGraph


class DShowVideoCamera(VideoInputStream):
    """Video camera input stream based on Microsoft DirectShow.
    """
    def __init__(self, source=0, format=0, is_color=True):
        self.source = source
        self.format = format
        self.is_color = is_color
        
        self.open()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_graph']
        del state['_current_frame']
        del state['_frame_grabbed']
        del state['_eof']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.open() 

    def __call__(self):        
        return self.read()
   
    def _on_grab_frame(self, frame):
        self._current_frame = frame
        self._frame_grabbed.set()

    @property
    def input_devices(self):
        return self._graph.get_input_devices()

    @property
    def formats(self):
        return self._graph.get_formats()
    
    def open(self):
        self._graph = FilterGraph()
        self._graph.add_input_device(self.source)
        self._graph.add_sample_grabber(self._on_grab_frame)
        self._graph.add_null_render()
        self._graph.set_format(self.format)
        self._graph.prepare()
        self._graph.run()

        self._current_frame = None
        self._frame_grabbed = threading.Event()
        self._eof = False        
            
    def read(self):   
        self._frame_grabbed.clear()
        self._graph.grab_frame()
        self._frame_grabbed.wait(10)
        if not self._frame_grabbed.is_set():
            self._eof = True
            raise EOFError
        if not self.is_color:
            self._current_frame = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2GRAY)
        return self._current_frame 
   
    def eof(self):
        return self._eof
    
    def close(self):
        self._graph.stop()
        