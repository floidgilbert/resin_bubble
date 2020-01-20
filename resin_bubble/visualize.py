import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import DatasetCatalog
from tqdm import tqdm

from .data import get_image_by_file_title, list_sort_images
from .transform import EqualizeTransform

__all__ = ['create_video_from_stills', 'plot_predictions', 'visualize_masks', 'visualize_metadata', 'VIZ_JPEG_QUALITY']

VIZ_JPEG_QUALITY = 40


def create_video_from_stills(input_directory: str, output_file: str) -> None:
    """
    Creates a video from a set of still images ordered by file name. For troubleshooting, refer to
    https://answers.opencv.org/question/66545/problems-with-the-video-writer-in-opencv-300/
    """

    def put_file_title(a: np.ndarray, filename: str, x: int = 5, y: int = 5,
                       scale: int = 1.5, thickness: int = 2) -> np.ndarray:
        file_title = os.path.splitext(os.path.basename(filename))[0]
        size = cv2.getTextSize(file_title, cv2.FONT_HERSHEY_PLAIN, scale, thickness)[0]
        a = cv2.rectangle(a, (x, y), (size[0] + x, size[1] + y), (255, 255, 255), -1)
        return cv2.putText(a, file_title, (x, size[1] + y), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness)

    files = list_sort_images(input_directory)
    image = cv2.imread(files[0])
    # Note width and height need to be backwards.
    vw = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 2., (image.shape[1], image.shape[0]))
    vw.write(put_file_title(image, files[0]))
    if not vw.isOpened():
        raise Exception('VideoWriter failed to initialize.')
    try:
        for file in tqdm(files[1:], desc=os.path.basename(output_file)):
            vw.write(put_file_title(cv2.imread(file), file))
    finally:
        vw.release()


def plot_predictions(info: dict, output_directory: str, figure_size: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a variety of plots for bubble count and area.
    """

    filenames = list(info.keys())
    filenames.sort()
    records = np.ndarray((len(filenames), 7), dtype=np.int32)
    for i, filename in enumerate(filenames):
        areas = np.array([instance['area'] for instance in info[filename]['instances']], dtype=records.dtype)
        percentiles = np.percentile(areas, (0, 25, 50, 75, 100)).astype(records.dtype)
        records[i] = np.array([[areas.shape[0], areas.sum(), *percentiles]], dtype=records.dtype)

    save_facecolor = plt.rcParams['axes.facecolor']
    plt.rcParams['axes.facecolor'] = (0.95, 0.95, 0.95)

    # Counts
    plt.figure(figsize=figure_size)
    plt.title('Bubble Count')
    plt.plot(records[:, 0])
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.grid(True, 'both')
    plt.savefig(os.path.join(output_directory, 'plot-counts.png'))

    # Area
    plt.figure(figsize=figure_size)
    plt.title('Total Bubble Area')
    plt.plot(records[:, 1])
    plt.xlabel('Frame')
    plt.ylabel('Area (Pixels)')
    plt.grid(True, 'both')
    plt.savefig(os.path.join(output_directory, 'plot-area-total.png'))

    # Area percentiles
    plt.figure(figsize=figure_size)
    plt.title('Bubble Area Quantiles')
    # plt.plot(records[:, 2])
    plt.plot(records[:, 5])
    plt.plot(records[:, 4])
    plt.plot(records[:, 3])
    # plt.plot(records[:, 6])
    plt.xlabel('Frame')
    plt.ylabel('Area (Pixels)')
    plt.legend(['3rd Quantile (75%)', 'Median (50%)', '1st Quantile (25%)'])
    plt.grid(True, 'both')
    plt.savefig(os.path.join(output_directory, 'plot-area-quantiles.png'))

    plt.rcParams['axes.facecolor'] = save_facecolor


def visualize_masks(
        mask_directory: str,
        image_directory: str,
        output_directory: str,
        info: Optional[dict] = None,
        tqdm_desc: str = 'Viz',
):
    mask_files = list_sort_images(mask_directory)
    os.makedirs(output_directory, exist_ok=True)
    eqt = EqualizeTransform()
    for mask_file in tqdm(mask_files, desc=tqdm_desc):
        mask_file_basename = os.path.basename(mask_file)
        mask_file_title = os.path.splitext(mask_file_basename)[0]
        image_file = get_image_by_file_title(image_directory, mask_file_title, raise_error=True)
        image = cv2.imread(image_file)
        image = eqt.apply_image(image)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        viz = np.zeros(image.shape, np.uint8)
        viz[..., 2] = mask
        viz = cv2.addWeighted(viz, 0.15, image, 1, 0)
        if info:
            file_dict = info[mask_file_basename]
            for instance in file_dict['instances']:
                viz = cv2.rectangle(viz, tuple(instance['pt1']), tuple(instance['pt2']), (0, 0, 0), 1)
            for instance in file_dict['filtered_by_min_area']:
                # Colors are in BGR order
                viz = cv2.rectangle(viz, tuple(instance['pt1']), tuple(instance['pt2']), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_directory, mask_file_title + '.jpg'), viz, [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


def visualize_metadata(catalog_name: str, output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)
    for record in DatasetCatalog.get(catalog_name):
        image = cv2.imread(record['file_name'])
        polygons = [None] * len(record['annotations'])
        for i, annotation in enumerate(record['annotations']):
            polygons[i] = annotation['segmentation'][0].reshape(-1, 1, 2)
        viz = np.zeros(image.shape, np.uint8)
        umat = cv2.drawContours(viz[..., 2], polygons, -1, 255, -1)
        viz[..., 2] = umat.get()
        viz = cv2.addWeighted(viz, 0.05, image, 1, 0)
        file_title = os.path.splitext(os.path.basename(record['file_name']))[0]
        cv2.imwrite(os.path.join(output_directory, file_title + '.jpg'), viz, [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])
