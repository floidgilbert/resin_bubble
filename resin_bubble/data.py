import itertools
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
from detectron2.config import CfgNode
from detectron2.data import DatasetMapper
from detectron2.structures import BoxMode
from tqdm import tqdm

from .transform import EqualizeTransformGen, RandomGaussianBlurTransformGen

SUPPORTED_IMAGE_EXTS = ('.jpg', '.png', '.tif', '.tiff', '.bmp')

__all__ = ['get_image_by_file_title', 'list_sort_images', 'load_annotations_from_masks', 'load_annotations_from_via',
           'ResinDataMapper', 'SUPPORTED_IMAGE_EXTS']


def get_image_by_file_title(
        directory: str,
        file_title: str,
        image_exts: Tuple[str] = SUPPORTED_IMAGE_EXTS,
        raise_error: bool = False
) -> str:
    """
    Returns the first image file name in a directory given a file title (i.e., the file name without an extension).

    :param directory: The directory containing the file.
    :param file_title: The file name excluding the extension.
    :param image_exts: Possible image file extensions.
    :param raise_error: Raise error if file not found.
    :return: File name with extension.
    """
    file_path = os.path.join(directory, file_title)
    for ext in image_exts:
        file_name = file_path + ext
        if os.path.isfile(file_name):
            return file_name
    if raise_error:
        msg = f"""The directory '{directory}' does not contain an image file named '{file_title}' with one of 
{SUPPORTED_IMAGE_EXTS} extensions."""
        raise RuntimeError(msg)


def list_sort_images(directory: str, image_exts: Tuple[str] = SUPPORTED_IMAGE_EXTS) -> List[str]:
    """
    Lists image files in a directory. The returned list is sorted alphabetically.

    :param directory: The target directory.
    :param image_exts: Files with these extensions are listed. Wildcards are not supported.
    :return: A list of files including paths, sorted alphabetically.
    """
    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files if file.lower().endswith(image_exts)]
    files.sort()
    return files


def load_annotations_from_masks(image_directory: str, mask_directory: str) -> List[dict]:
    """
    Converts mask images to detectron2 annotations. Assumes mask file names are the same as the original source images.
    Contiguous masked areas are converted to instances for instance segmentation.
    """

    records = []
    image_filenames = list_sort_images(image_directory)
    mask_filenames = list_sort_images(mask_directory)
    for image_filename, mask_filename in tqdm(zip(image_filenames, mask_filenames), desc='Mask Anno'):
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape
        record = dict()
        record['file_name'] = image_filename
        record['height'] = height
        record['width'] = width

        # polygons is a list of (n x 1 x 2) arrays.
        polygons = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        annotations = []
        for polygon in polygons:
            if polygon.shape[0] > 2:
                x1, y1 = np.min(polygon[:, 0, 0]), np.min(polygon[:, 0, 1])
                x2, y2 = np.max(polygon[:, 0, 0]), np.max(polygon[:, 0, 1])
                annotations.append(
                    {
                        'bbox': [x1, y1, x2, y2],
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'segmentation': [polygon.flatten()],
                        'category_id': 0,
                        'iscrowd': 0
                    }
                )
        record['annotations'] = annotations
        records.append(record)
    return records


def load_annotations_from_via(image_directory: str, annotations_file: str) -> List[dict]:
    """
    Converts annotations from a VIA JSON export file to detectron2 format. Currently, only circles and polygons are
    supported.
    """

    with open(annotations_file) as f:
        annotations = json.load(f)

    records = []
    for file in tqdm(annotations.values(), desc='Via Anno'):
        filename = os.path.join(image_directory, file['filename'])
        height, width = cv2.imread(filename).shape[:2]
        record = dict()
        record['file_name'] = filename
        record['height'] = height
        record['width'] = width

        annotations = []
        polygon_sides = 15
        for region in file['regions']:
            shape = region['shape_attributes']
            if shape['name'] == 'circle':
                # Doesn't catch items that go off of image edge
                cx, cy, r = shape['cx'], shape['cy'], shape['r']
                polygon_segment_radians = np.pi * 2 / polygon_sides
                px = [np.sin(polygon_segment_radians * i) * r + cx for i in range(polygon_sides)]
                py = [np.cos(polygon_segment_radians * i) * r + cy for i in range(polygon_sides)]
            elif shape['name'] == 'polygon':
                px = shape['all_points_x']
                py = shape['all_points_y']
            else:
                raise ValueError('Unsupported region shape: ' + shape['name'])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))  # Converts [(x1, y1)...] to [x1, y1, ...]
            annotations.append(
                {
                    'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [poly],
                    'category_id': 0,
                    'iscrowd': 0
                }
            )
        record['annotations'] = annotations
        records.append(record)
    return records


class ResinDataMapper(DatasetMapper):
    """
    Superclass detectron2 DatasetMapper to add custom image transformations.
    """

    def __init__(self, cfg: CfgNode, is_train: bool = True):
        super().__init__(cfg, is_train)

        # Append transformation generators to existing list.
        self.tfm_gens += [
            EqualizeTransformGen(),
            RandomGaussianBlurTransformGen(),
        ]

