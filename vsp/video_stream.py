# -*- coding: utf-8 -*-
"""Video stream classes provide common interfaces to video inputs/sources (e.g.,
camera, file) and outputs/destinations (e.g., display, file).
"""

import os
from abc import ABC, abstractmethod
from threading import Lock

import cv2


CV_VIDEO_CAPTURE_APIS = {
    'ANY': cv2.CAP_ANY,
    'VFW': cv2.CAP_VFW,
    'V4L': cv2.CAP_V4L,
    'V4L2': cv2.CAP_V4L2,
    'FIREWIRE': cv2.CAP_FIREWIRE,
    'FIREWARE': cv2.CAP_FIREWARE,
    'IEEE1394': cv2.CAP_IEEE1394,
    'DC1394': cv2.CAP_DC1394,
    'CMU1394': cv2.CAP_CMU1394,
    'QT': cv2.CAP_QT,
    'UNICAP': cv2.CAP_UNICAP,
    'DSHOW': cv2.CAP_DSHOW,
    'PVAPI': cv2.CAP_PVAPI,
    'OPENNI': cv2.CAP_OPENNI,
    'OPENNI_ASUS': cv2.CAP_OPENNI_ASUS,
    'ANDROID': cv2.CAP_ANDROID,
    'XIAPI': cv2.CAP_XIAPI,
    'AVFOUNDATION': cv2.CAP_AVFOUNDATION,
    'GIGANETIX': cv2.CAP_GIGANETIX,
    'MSMF': cv2.CAP_MSMF,
    'WINRT': cv2.CAP_WINRT,
    'INTELPERC': cv2.CAP_INTELPERC,
    'OPENNI2': cv2.CAP_OPENNI2,
    'OPENNI2_ASUS': cv2.CAP_OPENNI2_ASUS,
    'GPHOTO2': cv2.CAP_GPHOTO2,
    'GSTREAMER': cv2.CAP_GSTREAMER,
    'FFMPEG': cv2.CAP_FFMPEG,
    'IMAGES': cv2.CAP_IMAGES,
    'ARAVIS': cv2.CAP_ARAVIS,
    'OPENCV_MJPEG': cv2.CAP_OPENCV_MJPEG,
    'INTEL_MFX': cv2.CAP_INTEL_MFX,
    'XINE': cv2.CAP_XINE,
}

CV_VIDEO_CAPTURE_PROPERTIES = {
    'PROP_POS_MSEC': cv2.CAP_PROP_POS_MSEC,
    'PROP_POS_FRAMES': cv2.CAP_PROP_POS_FRAMES,
    'PROP_POS_AVI_RATIO': cv2.CAP_PROP_POS_AVI_RATIO,
    'PROP_FRAME_WIDTH': cv2.CAP_PROP_FRAME_WIDTH,
    'PROP_FRAME_HEIGHT': cv2.CAP_PROP_FRAME_HEIGHT,
    'PROP_FPS': cv2.CAP_PROP_FPS,
    'PROP_FOURCC': cv2.CAP_PROP_FOURCC,
    'PROP_FRAME_COUNT': cv2.CAP_PROP_FRAME_COUNT,
    'PROP_FORMAT': cv2.CAP_PROP_FORMAT,
    'PROP_MODE': cv2.CAP_PROP_MODE,
    'PROP_BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
    'PROP_CONTRAST': cv2.CAP_PROP_CONTRAST,
    'PROP_SATURATION': cv2.CAP_PROP_SATURATION,
    'PROP_HUE': cv2.CAP_PROP_HUE,
    'PROP_GAIN': cv2.CAP_PROP_GAIN,
    'PROP_EXPOSURE': cv2.CAP_PROP_EXPOSURE,
    'PROP_CONVERT_RGB': cv2.CAP_PROP_CONVERT_RGB,
    'PROP_WHITE_BALANCE_BLUE_U': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
    'PROP_RECTIFICATION': cv2.CAP_PROP_RECTIFICATION,
    'PROP_MONOCHROME': cv2.CAP_PROP_MONOCHROME,
    'PROP_SHARPNESS': cv2.CAP_PROP_SHARPNESS,
    'PROP_AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
    'PROP_GAMMA': cv2.CAP_PROP_GAMMA,
    'PROP_TEMPERATURE': cv2.CAP_PROP_TEMPERATURE,
    'PROP_TRIGGER': cv2.CAP_PROP_TRIGGER,
    'PROP_TRIGGER_DELAY': cv2.CAP_PROP_TRIGGER_DELAY,
    'PROP_WHITE_BALANCE_RED_V': cv2.CAP_PROP_WHITE_BALANCE_RED_V,
    'PROP_ZOOM': cv2.CAP_PROP_ZOOM,
    'PROP_FOCUS': cv2.CAP_PROP_FOCUS,
    'PROP_GUID': cv2.CAP_PROP_GUID,
    'PROP_ISO_SPEED': cv2.CAP_PROP_ISO_SPEED,
    'PROP_BACKLIGHT': cv2.CAP_PROP_BACKLIGHT,
    'PROP_PAN': cv2.CAP_PROP_PAN,
    'PROP_TILT': cv2.CAP_PROP_TILT,
    'PROP_ROLL': cv2.CAP_PROP_ROLL,
    'PROP_IRIS': cv2.CAP_PROP_IRIS,
    'PROP_SETTINGS': cv2.CAP_PROP_SETTINGS,
    'PROP_BUFFERSIZE': cv2.CAP_PROP_BUFFERSIZE,
    'PROP_AUTOFOCUS': cv2.CAP_PROP_AUTOFOCUS,
    'PROP_SAR_NUM': cv2.CAP_PROP_SAR_NUM,
    'PROP_SAR_DEN': cv2.CAP_PROP_SAR_DEN,
    'PROP_BACKEND': cv2.CAP_PROP_BACKEND,
    'PROP_CHANNEL': cv2.CAP_PROP_CHANNEL,
    'PROP_AUTO_WB': cv2.CAP_PROP_AUTO_WB,
    'PROP_WB_TEMPERATURE': cv2.CAP_PROP_WB_TEMPERATURE,
    # 'PROP_CODEC_PIXEL_FORMAT': cv2.CAP_PROP_CODEC_PIXEL_FORMAT,   # opencv-contrib
    # 'PROP_BITRATE': cv2.CAP_PROP_BITRATE,                         # opencv-contrib
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

    def grab(self):
        if not self._cap.isOpened():
            raise EOFError
        ret = self._cap.grab()
        if not ret:
            raise EOFError

    def retrieve(self):
        ret, frame = self._cap.retrieve()
        if not ret:
            raise EOFError
   
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

    def grab(self):
        if not self._cap.isOpened():
            raise EOFError
        ret = self._cap.grab()
        if not ret:
            raise EOFError

    def retrieve(self):
        ret, frame = self._cap.retrieve()
        if not ret:
            raise EOFError

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
    def __init__(self, filename="default.jpg"):
        self.filename = filename
        self.frame_id = 0

    def __call__(self, frame):
        return self.write(frame)

    def open(self):
        self.filename_root, self.filename_ext = os.path.splitext(self.filename)
        self.frame_id = 0

    def write(self, frame):
        cv2.imwrite(self.filename_root + '_' + str(self.frame_id) + self.filename_ext, frame)
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
