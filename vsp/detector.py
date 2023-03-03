# -*- coding: utf-8 -*-
"""Detector classes for detecting features in images or video frames.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from skimage import img_as_float
from skimage.feature import blob_doh
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.morphology import medial_axis

from vsp.feature import Keypoint
from vsp.optimizer import cross_entropy_optimizer


class Detector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass


class CvBlobDetector(Detector):
    """OpenCV blob detector detects blob keypoints in images or video frames.
    """

    def __init__(self,
                 threshold_step=None,
                 min_threshold=None,
                 max_threshold=None,
                 min_repeatability=None,
                 min_dist_between_blobs=None,
                 filter_by_color=None,
                 blob_color=None,
                 filter_by_area=None,
                 min_area=None,
                 max_area=None,
                 filter_by_circularity=None,
                 min_circularity=None,
                 max_circularity=None,
                 filter_by_inertia=None,
                 min_inertia_ratio=None,
                 max_inertia_ratio=None,
                 filter_by_convexity=None,
                 min_convexity=None,
                 max_convexity=None,
                 ):
        self.threshold_step = threshold_step
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_repeatability = min_repeatability
        self.min_dist_between_blobs = min_dist_between_blobs
        self.filter_by_color = filter_by_color
        self.blob_color = blob_color
        self.filter_by_area = filter_by_area
        self.min_area = min_area
        self.max_area = max_area
        self.filter_by_circularity = filter_by_circularity
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.filter_by_inertia = filter_by_inertia
        self.min_inertia_ratio = min_inertia_ratio
        self.max_inertia_ratio = max_inertia_ratio
        self.filter_by_convexity = filter_by_convexity
        self.min_convexity = min_convexity
        self.max_convexity = max_convexity

        self._init()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_detector']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init()

    def __call__(self, frame):
        return self.detect(frame)

    def _init(self):
        params = cv2.SimpleBlobDetector_Params()

        if self.threshold_step is not None:
            params.thresholdStep = self.threshold_step
        if self.min_threshold is not None:
            params.minThreshold = self.min_threshold
        if self.max_threshold is not None:
            params.maxThreshold = self.max_threshold
        if self.min_repeatability is not None:
            params.minRepeatability = self.min_repeatability
        if self.min_dist_between_blobs is not None:
            params.minDistBetweenBlobs = self.min_dist_between_blobs
        if self.filter_by_color is not None:
            params.filterByColor = self.filter_by_color
        if self.blob_color is not None:
            params.blobColor = self.blob_color
        if self.filter_by_area is not None:
            params.filterByArea = self.filter_by_area
        if self.min_area is not None:
            params.minArea = self.min_area
        if self.max_area is not None:
            params.maxArea = self.max_area
        if self.filter_by_circularity is not None:
            params.filterByCircularity = self.filter_by_circularity
        if self.min_circularity is not None:
            params.minCircularity = self.min_circularity
        if self.max_circularity is not None:
            params.maxCircularity = self.max_circularity
        if self.filter_by_inertia is not None:
            params.filterByInertia = self.filter_by_inertia
        if self.min_inertia_ratio is not None:
            params.minInertiaRatio = self.min_inertia_ratio
        if self.max_inertia_ratio is not None:
            params.maxInertiaRatio = self.max_inertia_ratio
        if self.filter_by_convexity is not None:
            params.filterByConvexity = self.filter_by_convexity
        if self.min_convexity is not None:
            params.minConvexity = self.min_convexity
        if self.max_convexity is not None:
            params.maxConvexity = self.max_convexity

        self._detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame):
        keypoints_cv = self._detector.detect(frame)
        keypoints = [Keypoint(kp.pt, kp.size) for kp in keypoints_cv]
        return keypoints


def target_blobs_cost_func(frames,
                           target_blobs,
                           threshold_step=None,
                           min_threshold=None,
                           max_threshold=None,
                           min_repeatability=None,
                           min_dist_between_blobs=None,
                           filter_by_color=None,
                           blob_color=None,
                           filter_by_area=None,
                           min_area=None,
                           max_area=None,
                           filter_by_circularity=None,
                           min_circularity=None,
                           max_circularity=None,
                           filter_by_inertia=None,
                           min_inertia_ratio=None,
                           max_inertia_ratio=None,
                           filter_by_convexity=None,
                           min_convexity=None,
                           max_convexity=None,
                           ):
    """Simple cost function for optimizing Open CV blob detector.
    """
    params = dict(locals())
    del params['frames']
    del params['target_blobs']

    if min_threshold is not None and max_threshold is not None \
            and min_threshold > max_threshold:
        cost = np.inf
    elif min_area is not None and max_area is not None \
            and min_area > max_area:
        cost = np.inf
    elif min_circularity is not None and max_circularity is not None \
            and min_circularity > max_circularity:
        cost = np.inf
    elif min_inertia_ratio is not None and max_inertia_ratio is not None \
            and min_inertia_ratio > max_inertia_ratio:
        cost = np.inf
    elif min_convexity is not None and max_convexity is not None \
            and min_convexity > max_convexity:
        cost = np.inf
    else:
        cost = 0.0
        det = CvBlobDetector(**params)
        for frame in frames:
            kpts = det.detect(frame)
            detected_blobs = len(kpts)
            cost += abs(target_blobs - detected_blobs)
        cost /= frames.shape[0]

    return cost


def optimize_blob_detector_params(frames,
                                  target_blobs,
                                  min_threshold_range=(0, 300),
                                  max_threshold_range=(0, 300),
                                  min_area_range=(0, 200),
                                  max_area_range=(0, 200),
                                  min_circularity_range=(0.1, 0.9),
                                  min_inertia_ratio_range=(0.1, 0.9),
                                  min_convexity_range=(0.1, 0.9),
                                  blob_color=255,
                                  ):
    """Optimize Open CV blob detector parameters.
    """
    # if frames.ndim == 3:
    #     frames = frames[np.newaxis, ...]

    def func(x): return target_blobs_cost_func(frames,
                                               target_blobs,
                                               min_threshold=x[0],
                                               max_threshold=x[1],
                                               filter_by_color=True,
                                               blob_color=blob_color,
                                               filter_by_area=True,
                                               min_area=x[2],
                                               max_area=x[3],
                                               filter_by_circularity=True,
                                               min_circularity=x[4],
                                               filter_by_inertia=True,
                                               min_inertia_ratio=x[5],
                                               filter_by_convexity=True,
                                               min_convexity=x[6],
                                               )

    xrng = np.array((min_threshold_range,
                     max_threshold_range,
                     min_area_range,
                     max_area_range,
                     min_circularity_range,
                     min_inertia_ratio_range,
                     min_convexity_range,
                     ))

    xopt = cross_entropy_optimizer(func,
                                   xrng,
                                   pop_size=50,
                                   elite_size=10,
                                   max_iters=15)

    opt_params = {'min_threshold': xopt[0],
                  'max_threshold': xopt[1],
                  'filter_by_color': True,
                  'blob_color': 255,
                  'filter_by_area': True,
                  'min_area': xopt[2],
                  'max_area': xopt[3],
                  'filter_by_circularity': True,
                  'min_circularity': xopt[4],
                  'filter_by_inertia': True,
                  'min_inertia_ratio': xopt[5],
                  'filter_by_convexity': True,
                  'min_convexity': xopt[6],
                  }

    return opt_params


class CvContourBlobDetector(Detector):
    """OpenCV contour-based blob detector detects blob keypoints in images or video frames.
    """

    def __init__(self, blur_kernel_size=9, thresh_block_size=11, thresh_constant=-30.0, min_radius=4, max_radius=14):
        self.blur_kernel_size = blur_kernel_size
        self.thresh_block_size = thresh_block_size
        self.thresh_constant = thresh_constant
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, frame):
        return self.detect(frame)

    def detect(self, frame):

        # helps to smooth the image, removing artefacts
        frame = cv2.blur(frame, (self.blur_kernel_size, self.blur_kernel_size))

        # threshhold into binary image
        frame = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.thresh_block_size, self.thresh_constant
        )

        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        keypoints_cv = [cv2.minEnclosingCircle(c) for c in contours]
        keypoints = [Keypoint(kp[0], kp[1]) for kp in keypoints_cv if kp[1] >= self.min_radius
                     and kp[1] <= self.max_radius]
        return keypoints


def target_contour_blobs_cost_func(frames,
                                   target_blobs,
                                   blur_kernel_size,
                                   thresh_block_size,
                                   thresh_constant,
                                   min_radius,
                                   max_radius,
                                   ):
    """Simple cost function for optimizing Open CV blob detector.
    """
    params = dict(locals())
    del params['frames']
    del params['target_blobs']

    if min_radius is not None and max_radius is not None \
            and int(min_radius) >= int(max_radius):
        cost = np.inf
    else:
        cost = 0.0
        det = CvContourBlobDetector(blur_kernel_size, thresh_block_size, thresh_constant, min_radius, max_radius)
        for frame in frames:
            kpts = det.detect(frame)
            detected_blobs = len(kpts)
            cost += abs(target_blobs - detected_blobs)
        cost /= frames.shape[0]

    return cost


def optimize_contour_blob_detector_params(frames,
                                          target_blobs,
                                          blur_kernel_size_half_range,
                                          thresh_block_size_half_range,
                                          thresh_constant_range,
                                          min_radius_range,
                                          max_radius_range,
                                          ):
    """Optimize Open CV blob detector parameters.
    """
    # if frames.ndim == 3:
    #     frames = frames[np.newaxis, ...]

    def func(x): return target_contour_blobs_cost_func(frames,
                                                       target_blobs,
                                                       blur_kernel_size=2 * int(x[0]) + 1,
                                                       thresh_block_size=2 * int(x[1]) + 1,
                                                       thresh_constant=int(x[2]),
                                                       min_radius=int(x[3]),
                                                       max_radius=int(x[4]),
                                                       )

    xrng = np.array((blur_kernel_size_half_range,
                     thresh_block_size_half_range,
                     thresh_constant_range,
                     min_radius_range,
                     max_radius_range,
                     ))

    xopt = cross_entropy_optimizer(func,
                                   xrng,
                                   pop_size=50,
                                   elite_size=10,
                                   max_iters=25)

    opt_params = {'blur_kernel_size': 2 * int(xopt[0]) + 1,
                  'thresh_block_size': 2 * int(xopt[1]) + 1,
                  'thresh_constant': int(xopt[2]),
                  'min_radius': int(xopt[3]),
                  'max_radius': int(xopt[4]),
                  }

    return opt_params


class SklDoHBlobDetector(Detector):
    """Scikit-Learn Determinant-of-Hessian blob detector detects blob keypoints in images or video frames.
    """

    def __init__(self, min_sigma=5.0, max_sigma=6.0, num_sigma=5, threshold=0.015):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold

    def __call__(self, frame):
        return self.detect(frame)

    def detect(self, frame):
        sk_frame = img_as_float(frame)
        if len(frame.shape) == 3:
            sk_frame = rgb2gray(sk_frame)
        blobs = blob_doh(sk_frame, min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                         num_sigma=self.num_sigma, threshold=self.threshold)
        keypoints = [Keypoint((x, y), r) for y, x, r in blobs]
        return keypoints


def doh_blobs_cost_func(frames,
                        target_blobs,
                        min_sigma,
                        max_sigma,
                        num_sigma,
                        threshold,
                        ):
    """Simple cost function for optimizing Scikit DoH blob detector.
    """
    params = dict(locals())
    del params['frames']
    del params['target_blobs']

    cost = 0.0
    det = SklDoHBlobDetector(
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )
    for frame in frames:
        kpts = det.detect(frame)
        detected_blobs = len(kpts)
        cost += abs(target_blobs - detected_blobs)
    cost /= frames.shape[0]

    return cost


def optimize_doh_blob_detector_params(frames,
                                      target_blobs,
                                      min_sigma_range,
                                      max_sigma_range,
                                      num_sigma_range,
                                      threshold_range
                                      ):
    """Optimize Scikit DoH detector parameters.
    """

    def func(x): return doh_blobs_cost_func(frames,
                                            target_blobs,
                                            min_sigma=x[0],
                                            max_sigma=x[1],
                                            num_sigma=int(x[2]),
                                            threshold=x[3]
                                            )

    xrng = np.array((min_sigma_range,
                     max_sigma_range,
                     num_sigma_range,
                     threshold_range
                     ))

    xopt = cross_entropy_optimizer(func,
                                   xrng,
                                   pop_size=50,
                                   elite_size=10,
                                   max_iters=15)

    opt_params = {'min_sigma': xopt[0],
                  'max_sigma': xopt[1],
                  'num_sigma': int(xopt[2]),
                  'threshold': xopt[3]
                  }

    return opt_params


class SkeletonizePeakDetector(Detector):
    """Extract keypoints as peaks from skeletonized images.
    """

    def __init__(
        self,
        blur_kernel_size=9,
        min_distance=10,
        threshold_abs=0.4346,
        num_peaks=331,
        thresh_block_size=11,
        thresh_constant=-34.0,
    ):
        self.blur_kernel_size = blur_kernel_size
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.num_peaks = num_peaks
        self.thresh_block_size = thresh_block_size
        self.thresh_constant = thresh_constant
        self.kp_size_multiplier = 3.0

    def __call__(self, frame):
        return self.detect(frame)

    def detect(self, frame):

        # helps to smooth the image, removing artefacts
        frame = cv2.blur(frame, (self.blur_kernel_size, self.blur_kernel_size))

        # threshhold into binary image
        frame = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.thresh_block_size, self.thresh_constant
        )

        # skeletonize the image to get central peaks
        skel, distance = medial_axis(frame, return_distance=True)
        skel_frame = distance * skel

        # extract keypoints as local peaks in skeletonized image
        keypoints = peak_local_max(skel_frame, min_distance=self.min_distance,
                                   threshold_abs=self.threshold_abs, num_peaks=self.num_peaks)
        size = distance[keypoints[:, 0], keypoints[:, 1]]

        return [Keypoint((float(x), float(y)), r * self.kp_size_multiplier) for (y, x), r in zip(keypoints, size)]


def skeltonize_peaks_cost_func(frames,
                               target_blobs,
                               blur_kernel_size,
                               min_distance,
                               threshold_abs,
                               thresh_block_size,
                               thresh_constant,
                               ):
    """Simple cost function for optimizing skeletonize blob detector.
    """
    params = dict(locals())
    del params['frames']
    del params['target_blobs']

    cost = 0.0
    det = SkeletonizePeakDetector(
        blur_kernel_size=blur_kernel_size,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        num_peaks=np.inf,
        thresh_block_size=thresh_block_size,
        thresh_constant=thresh_constant
    )
    for frame in frames:
        kpts = det.detect(frame)
        detected_blobs = len(kpts)
        cost += abs(target_blobs - detected_blobs)
    cost /= frames.shape[0]

    return cost


def optimize_skeltonize_peak_detector_params(frames,
                                             target_blobs,
                                             blur_kernel_size_half_range,
                                             min_distance_range,
                                             threshold_abs_range,
                                             thresh_block_size_half_range,
                                             thresh_constant_range,
                                             ):
    """Optimize skeletonize blob detector parameters.
    """

    def func(x): return skeltonize_peaks_cost_func(frames,
                                                   target_blobs,
                                                   blur_kernel_size=2 * int(x[0]) + 1,
                                                   min_distance=int(x[1]),
                                                   threshold_abs=x[2],
                                                   thresh_block_size=2 * int(x[3]) + 1,
                                                   thresh_constant=x[4],
                                                   )

    xrng = np.array((blur_kernel_size_half_range,
                     min_distance_range,
                     threshold_abs_range,
                     thresh_block_size_half_range,
                     thresh_constant_range,
                     ))

    xopt = cross_entropy_optimizer(func,
                                   xrng,
                                   pop_size=50,
                                   elite_size=10,
                                   max_iters=15)

    opt_params = {'blur_kernel_size': 2 * int(xopt[0]) + 1,
                  'min_distance': int(xopt[1]),
                  'threshold_abs': xopt[2],
                  'thresh_block_size': 2 * int(xopt[3]) + 1,
                  'thresh_constant': xopt[4],
                  }

    return opt_params
