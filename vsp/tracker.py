# -*- coding: utf-8 -*-
"""Tracker classes for trackimg features in images or video frames.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np
import scipy.spatial.distance as ssd

from vsp.feature import Keypoint


class KeypointTypeError(TypeError):
    pass


def to_keypoint_array(keypoints):
    if isinstance(keypoints, np.ndarray) and isinstance(keypoints[0], Keypoint):
        # already keypoint array
        return keypoints

    if isinstance(keypoints, list) and isinstance(keypoints[0], Keypoint):
        # convert keypoint list to keypoint array
        return np.array(keypoints)

    if isinstance(keypoints, np.ndarray) and not isinstance(keypoints[0], Keypoint):
        # convert numpy array to keypoint array
        return np.array([Keypoint(kp[:2], *kp[2:]) for kp in keypoints])

    raise KeypointTypeError


class Tracker(ABC):
    @abstractmethod
    def track(self, keypoints):
        pass


class NearestNeighbourTracker(Tracker):
    """Tracks features in images or video frames using nearest-neighbour tracking.
    """
    def __init__(self, threshold, keypoints=None):
        self.threshold = threshold
        self.keypoints = keypoints
        if self.keypoints is not None:
            self.keypoints = to_keypoint_array(self.keypoints)

    def __call__(self, keypoints):        
        return self.track(keypoints)

    def track(self, keypoints):
        if self.keypoints is None:
            self.keypoints = np.array(keypoints)
        else:            
            # Map keypoints to closest previous keypoints as long as they are
            # closer than the threshold distance; otherwise use the previous
            # ones
            keypoints = np.array(keypoints)
            points = np.array([kp.point for kp in keypoints])
            prev_points = np.array([kp.point for kp in self.keypoints])
            dists = ssd.cdist(points, prev_points, 'euclidean')
            min_dists = np.min(dists, axis=1)
            min_dist_idxs = np.argmin(dists, axis=1)
            replace_keypoints = keypoints[min_dists < self.threshold]
            replace_idxs = min_dist_idxs[min_dists < self.threshold]
            self.keypoints[replace_idxs] = replace_keypoints
        return self.keypoints.tolist()

    def reset(self, keypoints=None):
        self.keypoints = keypoints
        if self.keypoints is not None:
            self.keypoints = to_keypoint_array(self.keypoints)
