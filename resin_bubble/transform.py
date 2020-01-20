import cv2
import numpy as np
from detectron2.data.transforms import NoOpTransform, Transform, TransformGen
from PIL import Image
from PIL.ImageOps import equalize

__all__ = ['EqualizeTransform', 'EqualizeTransformGen', 'RandomGaussianBlurTransform',
           'RandomGaussianBlurTransformGen']


class EqualizeTransform(Transform):
    # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

    def apply_coords(self, coords: np.ndarray):
        return coords

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return np.array(equalize(Image.fromarray(img)))


class EqualizeTransformGen(TransformGen):

    def get_transform(self, img: np.ndarray) -> EqualizeTransform:
        return EqualizeTransform()


class RandomGaussianBlurTransform(Transform):

    def __init__(self, ksize=(7, 7), sigmaX=1, sigmaY=1):
        super().__init__()
        self._set_attributes(locals())

    def apply_coords(self, coords: np.ndarray):
        return coords

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, ksize=self.ksize, sigmaX=self.sigmaX, sigmaY=self.sigmaY)


class RandomGaussianBlurTransformGen(TransformGen):

    def __init__(self, ksize=(7, 7), sigmaX=1, sigmaY=1):
        super().__init__()
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if np.random.random_sample() < 0.5:
            return RandomGaussianBlurTransform(ksize=self.ksize, sigmaX=self.sigmaX, sigmaY=self.sigmaY)
        return NoOpTransform()
