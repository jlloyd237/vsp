# -*- coding: utf-8 -*-
"""Encoder classes for encoding features (feature keypoints and/or descriptors).
"""

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    @abstractmethod
    def encode(self, keypoints):
        pass


class KeypointEncoder(Encoder):
    def __call__(self, keypoints):        
        return self.encode(keypoints)
		
    def encode(self, keypoints):
        return np.array([kp.point + (kp.size,) for kp in keypoints])


class VoronoiEncoder(Encoder):
    pass


class ChannelEncoder(Encoder):
    pass
