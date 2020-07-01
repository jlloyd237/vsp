# -*- coding: utf-8 -*-
"""Feature classes for representing detected features.

   An image feature (patch) is composed of a feature keypoint and a feature
   descriptor. The keypoint contains the 2D position of the patch and other
   attributes if available, such as scale and orientation of the image feature.
   The descriptor contains the visual description of the patch and is used to
   compare the similarity between image features.
"""

import collections

import numpy as np

Point = collections.namedtuple('Point', 'x y')   

class Keypoint:
    """Feature keypoint.
    """
    def __init__(self, point, size=0.0, angle=0.0):
        self.point = np.asarray(point)
        self.size = size
        self.angle = angle       
   