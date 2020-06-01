# -*- coding: utf-8 -*-
"""Video stream classes provide common interfaces to video inputs/sources (e.g.,
camera, file) and outputs/destinations (e.g., display, file).
"""

import os
from abc import ABC, abstractmethod
from threading import Lock

import cv2


CV_VIDEO_CAPTURE_APIS = {
    'ANY': 0,
    'VFW': 200,
    'V4L': 200,
    'V4L2': 200,
    'FIREWIRE': 300,
    'FIREWARE': 300,
    'IEEE1394': 300,
    'DC1394': 300,
    'CMU1394': 300,
    'QT': 500,
    'UNICAP': 600,
    'DSHOW': 700,
    'PVAPI': 800,
    'OPENNI': 900,
    'OPENNI_ASUS': 910,
    'ANDROID': 1000,
    'XIAPI': 1100,
    'AVFOUNDATION': 1200,
    'GIGANETIX': 1300,
    'MSMF': 1400,
    'WINRT': 1410,
    'INTELPERC': 1500,
    'OPENNI2': 1600,
    'OPENNI2_ASUS': 1610,
    'GPHOTO2': 1700,
    'GSTREAMER': 1800,
    'FFMPEG': 1900,
    'IMAGES': 2000,
    'ARAVIS': 2100,
    'OPENCV_MJPEG': 2200,
    'INTEL_MFX': 2300,
    'XINE': 2400,
}

CV_VIDEO_CAPTURE_PROPERTIES = {
    'PROP_POS_MSEC': 0,
    'PROP_POS_FRAMES': 1,
    'PROP_POS_AVI_RATIO': 2,
    'PROP_FRAME_WIDTH': 3,
    'PROP_FRAME_HEIGHT': 4,
    'PROP_FPS': 5,
    'PROP_FOURCC': 6,
    'PROP_FRAME_COUNT': 7,
    'PROP_FORMAT': 8,
    'PROP_MODE': 9,
    'PROP_BRIGHTNESS': 10,
    'PROP_CONTRAST': 11,
    'PROP_SATURATION': 12,
    'PROP_HUE': 13,
    'PROP_GAIN': 14,
    'PROP_EXPOSURE': 15,
    'PROP_CONVERT_RGB': 16,
    'PROP_WHITE_BALANCE_BLUE_U': 17,
    'PROP_RECTIFICATION': 18,
    'PROP_MONOCHROME': 19,
    'PROP_SHARPNESS': 20,
    'PROP_AUTO_EXPOSURE': 21,
    'PROP_GAMMA': 22,
    'PROP_TEMPERATURE': 23,
    'PROP_TRIGGER': 24,
    'PROP_TRIGGER_DELAY': 25,
    'PROP_WHITE_BALANCE_RED_V': 26,
    'PROP_ZOOM': 27,
    'PROP_FOCUS': 28,
    'PROP_GUID': 29,
    'PROP_ISO_SPEED': 30,
    'PROP_BACKLIGHT': 32,
    'PROP_PAN': 33,
    'PROP_TILT': 34,
    'PROP_ROLL': 35,
    'PROP_IRIS': 36,
    'PROP_SETTINGS': 37,
    'PROP_BUFFERSIZE': 38,
    'PROP_AUTOFOCUS': 39,
    'PROP_SAR_NUM': 40,
    'PROP_SAR_DEN': 41,
    'PROP_BACKEND': 42,
    'PROP_CHANNEL': 43,
    'PROP_AUTO_WB': 44,
    'PROP_WB_TEMPERATURE': 45,
    'PROP_CODEC_PIXEL_FORMAT': 46,
    'PROP_BITRATE': 47,
}

class UnknownAPIError(ValueError):
    pass

class UnknownPropertyError(ValueError):
    pass

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
                 api_name=None,
                 frame_size=None,
                 brightness=None,
                 contrast=None,
                 exposure=None,
                 is_color=True,
                 ):
        self.source = source
        self.api_name = api_name
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

    @property
    def camera_api(self):
        return self._cap.getBackendName()

    def get_property(self, prop_name):
        prop_id = CV_VIDEO_CAPTURE_PROPERTIES.get(prop_name, None)
        if prop_id is None:
            raise UnknownPropertyError()
        return self._cap.get(prop_id)

    def set_property(self, prop_name, prop_val):
        prop_id = CV_VIDEO_CAPTURE_PROPERTIES.get(prop_name, None)
        if prop_id is None:
            raise UnknownPropertyError()
        return self._cap.set(prop_id, prop_val)

    def open(self):
        if self.api_name is not None:
            api_id = CV_VIDEO_CAPTURE_APIS.get(self.api_name, None)
            if api_id is None:
                raise UnknownAPIError()
            self._cap = cv2.VideoCapture(self.source, api_id)
        else:
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
    def __init__(self, filename="default.mp4", api_name=None, is_color=True):
        self.filename = filename
        self.api_name = api_name
        self.is_color = is_color

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_capture' in state:
            del state['_cap']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.open()

    def __call__(self):        
        return self.read()

    @property
    def camera_api(self):
        return self._cap.getBackendName()

    def get_property(self, prop_name):
        prop_id = CV_VIDEO_CAPTURE_PROPERTIES.get(prop_name, None)
        if prop_id is None:
            raise UnknownPropertyError()
        return self._cap.get(prop_id)

    def set_property(self, prop_name, prop_val):
        prop_id = CV_VIDEO_CAPTURE_PROPERTIES.get(prop_name, None)
        if prop_id is None:
            raise UnknownPropertyError()
        return self._cap.set(prop_id, prop_val)

    def open(self):
        if self.api_name is not None:
            api_id = CV_VIDEO_CAPTURE_APIS.get(self.api_name, None)
            if api_id is None:
                raise UnknownAPIError()
            self._cap = cv2.VideoCapture(self.filename, api_id)
        else:
            self._cap = cv2.VideoCapture(self.filename)
            
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

    def __init__(self, name="default"):
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
                 filename="default.mp4",
                 fourcc_code='mp4v',
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


class CvImageOutputFileSeq(VideoOutputStream):
    """OpenCV image output file sequence.
    """
    def __init__(self, filename="default.jpg", start_frame=0):
        self.filename = filename
        self.start_frame = start_frame
        self.frame_id = 0

    def __call__(self, frame):
        return self.write(frame)

    def open(self):
        self.filename_root, self.filename_ext = os.path.splitext(self.filename)
        self.frame_id = 0

    def write(self, frame):
        if self.frame_id >= self.start_frame:
            cv2.imwrite(self.filename_root + '_' + str(self.frame_id - self.start_frame) + self.filename_ext, frame)
        self.frame_id += 1

    def close(self):
        self.frame_id = 0


class CvImageInputFileSeq(VideoInputStream):
    """OpenCV image input file sequence.
    """

    def __init__(self, filename="default.jpg"):
        # For filename="root.ext", image files should be named "root_0.ext", "root_1.ext", ...
        self.filename = filename
        self.frame_id = 0

    def __call__(self):
        return self.read()

    def open(self):
        self.filename_root, self.filename_ext = os.path.splitext(self.filename)
        self.frame_id = 0

    def read(self):
        frame = cv2.imread(self.filename_root + '_' + str(self.frame_id) + self.filename_ext,
                           cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise EOFError
        self.frame_id += 1
        return frame

    def eof(self):
        return False

    def close(self):
        self.frame_id = 0
