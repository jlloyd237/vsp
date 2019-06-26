# -*- coding: utf-8 -*-
"""View classes for overlaying objects (e.g., keypoints) on images and video frames.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from vsp.feature import Keypoint


class View(ABC):
    @abstractmethod
    def draw(frame, *objs):
        pass


class KeypointView(View):
    """Overlays a list, tuple or array of keypoints on a video frame, using coloured circles to
    represent the location and size of each keypoint.
    """
    def __init__(self, color=(0,0,255)):
        self.color = color

    def __call__(self, frame, keypoints):        
        return self.draw(frame, keypoints)
		
    def draw(self, frame, keypoints):
        if isinstance(keypoints, (list, tuple, np.ndarray)) \
            and isinstance(keypoints[0], Keypoint):
            keypoints_cv = [cv2.KeyPoint(kp.point[0], kp.point[1], kp.size) for kp in keypoints]      
        elif isinstance(keypoints, np.ndarray):
            keypoints_cv = [cv2.KeyPoint(kp[0], kp[1], kp[2]) for kp in keypoints]
        else:
            raise TypeError("Unknown keypoints type")
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints_cv, np.array([]), self.color,
                                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame_with_keypoints
