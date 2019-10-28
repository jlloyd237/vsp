# -*- coding: utf-8 -*-
"""Video stream classes provide common interfaces to video inputs/sources (e.g.,
camera, file) and outputs/destinations (e.g., display, file).
"""

from abc import ABC, abstractmethod
from threading import Lock

import cv2


class VideoInputStream(ABC):
    """Video input stream provides a common interface for different video
    inputs/sources.
    """
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            item = self.read()
        except EOFError:
            raise StopIteration
        return item

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def read(self):
        pass
    
    @abstractmethod
    def eof(self):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    
class VideoOutputStream(ABC):
    """Video output stream provides a common interface for different video
    outputs/destinations.
    """
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close()

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def write(self):
        pass
    
    @abstractmethod
    def close(self):
        pass    
        
        
class CvVideoCamera(VideoInputStream):
    """OpenCV video camera input stream.
    """
    def __init__(self,
                 source=0,
                 frame_size=None,
                 brightness=None,
                 contrast=None,
                 exposure=None,
                 is_color=True,
                 ):
        self.source = source
        self.frame_size = frame_size
        self.brightness = brightness
        self.contrast = contrast
        self.exposure = exposure
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
    
    def open(self):
        self._cap = cv2.VideoCapture(self.source)            
        if self.frame_size is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])           
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        if self.brightness is not None:
            self._cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        if self.contrast is not None:
            self._cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        if self.exposure is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)              
            
    def read(self):
        if not self._cap.isOpened():
            raise EOFError
        ret, frame = self._cap.read()
        if not ret:
            raise EOFError
        if not self.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
   
    def eof(self):
        return not self._cap.isOpened()
    
    def close(self):
        self._cap.release()


class CvVideoInputFile(VideoInputStream):
    """OpenCV video input file.
    """
    def __init__(self, filename='default.mp4', is_color=True):
        self.filename = filename
        self.is_color = is_color

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_capture' in state:
            del state['_capture']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.open()

    def __call__(self):        
        return self.read()

    def open(self):
        self._capture = cv2.VideoCapture(self.filename)
            
    def read(self):
        if not self._capture.isOpened():
            raise EOFError
        ret, frame = self._capture.read()
        if not ret:
            raise EOFError
        if not self.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    
    def eof(self):
        return not self._capture.isOpened()
    
    def close(self):
        self._capture.release()


class CvVideoDisplay(VideoOutputStream):
    """OpenCV video display.
    """
    class __CvVideoDisplay(VideoOutputStream):
        def __init__(self):
            self._lock = Lock()
        def open(self, name):
            with self._lock:
                cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

        def write(self, name, frame):
            with self._lock:
                cv2.imshow(name, frame)
                cv2.waitKey(1)

        def close(self, name):
            with self._lock:
                cv2.destroyWindow(name)

    instance = None

    def __init__(self, name='default'):
        self.name = name

    def __call__(self, frame):
        return self.write(frame)

    def open(self):
        if not CvVideoDisplay.instance:
            CvVideoDisplay.instance = CvVideoDisplay.__CvVideoDisplay()
        CvVideoDisplay.instance.open(self.name)

    def write(self, frame):
        CvVideoDisplay.instance.write(self.name, frame)

    def close(self):
        CvVideoDisplay.instance.close(self.name)

   
class CvVideoOutputFile(VideoOutputStream):
    """OpenCV video output file.
    """
    def __init__(self,
                 filename='default.mp4',
                 fourcc_code='MP4V',
                 fps=30.0,
                 frame_size=(640, 480),
                 is_color=True,
                 ):
        self.filename = filename
        self.fourcc_code = fourcc_code
        self.fps = fps
        self.frame_size = frame_size
        self.is_color = is_color

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_writer' in state:
            del state['_writer']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.open()

    def __call__(self, frame):        
        return self.write(frame)
        
    def open(self):
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc_code)
        self._writer = cv2.VideoWriter(self.filename, fourcc, self.fps,
                                       self.frame_size, self.is_color)
        
    def write(self, frame):
        self._writer.write(frame)
    
    def close(self):
        self._writer.release()
