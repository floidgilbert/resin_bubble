import gc
import os
from typing import List, Union

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data.transforms import apply_transform_gens
from detectron2.modeling import build_model
from tqdm import tqdm

from .data import list_sort_images
from .transform import EqualizeTransformGen

__all__ = ['EnsembleBatchPredictor', 'get_masks_info', 'ResinPredictor']


def get_masks_info(mask_directory: str, instance_min_area: int) -> dict:
    """
    Summarizes instance information for predicted masks into the following dictionary format.

    {<mask file name>:
        {
            'height': <image height>
            'width': <image width>
            'total_instance_area': <total area of all instances>
            'instances': [
                {'area': <area>, 'pt1': (<min x>, <min y>), 'pt2': (<max x>, <max y>)},
                ...
            ]
            'filtered_by_min_area': {
                {'area': <area>, 'pt1': (<min x>, <min y>), 'pt2': (<max x>, <max y>)},
                ...
            }
    , ...
    }
    """

    mask_files = list_sort_images(mask_directory)
    records = {}
    for mask_file in mask_files:
        mask: np.ndarray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # polygons is a list of (n x 1 x 2) arrays.
        polygons = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        instances = []
        filtered_by_min_area = []
        total_instance_area = 0
        for polygon in polygons:
            if polygon.shape[0] > 2:
                x1, y1 = np.min(polygon[:, 0, 0]).item(), np.min(polygon[:, 0, 1]).item()
                x2, y2 = np.max(polygon[:, 0, 0]).item(), np.max(polygon[:, 0, 1]).item()
                temp = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
                temp = cv2.fillPoly(temp, [polygon], 1, offset=(-x1, -y1))
                area = temp.sum().item()
                instance = dict(area=area, pt1=[x1, y1], pt2=[x2, y2])
                if area >= instance_min_area:
                    instances.append(instance)
                    total_instance_area += area
                else:
                    filtered_by_min_area.append(instance)
        records[os.path.basename(mask_file)] = dict(
            height=mask.shape[0],
            width=mask.shape[1],
            total_instance_area=total_instance_area,
            instances=instances,
            filtered_by_min_area=filtered_by_min_area,
        )
    return records


class EnsembleBatchPredictor:
    """
    Predicts instances for multiple images using one or more PyTorch models. Writes mask images to the specified
    output directory.
    """

    def __init__(
            self,
            cfg: CfgNode,
            models: List[str],
    ):
        """
        :param cfg: Detectron2 CfgNode loaded from YAML file.
        :param models: File names specifying one or more PyTorch models to use for prediction.
        """
        self.cfg = cfg
        self.models = models

    def __call__(self, input_files: List[str], output_directory: str):
        """
        :param input_files: List of image files.
        :param output_directory: Path to directory where mask images are written.
        """

        # Each model adds to the existing mask if it exists. Create a list of mask files and delete any existing.
        os.makedirs(output_directory, exist_ok=True)
        mask_files: List[Union[str, None]] = [None] * len(input_files)
        for i, input_file in enumerate(input_files):
            mask_file = os.path.splitext(os.path.basename(input_file))[0] + '.png'
            mask_files[i] = os.path.join(output_directory, mask_file)
            if os.path.isfile(mask_files[i]):
                os.remove(mask_files[i])

        # For each model, merge instances into mask file.
        for model in self.models:
            self.cfg.MODEL.WEIGHTS = model
            model_basename = os.path.basename(model)
            print(f'Loading model \'{model_basename}\'...')
            predictor = ResinPredictor(self.cfg)
            # Retrieve all instances and save them to temporary files.
            for input_file, mask_file in tqdm(zip(input_files, mask_files), desc='Predicting', total=len(input_files)):
                image = cv2.imread(input_file)
                self.__merge_instances(
                    predictor(image)['instances'].get_fields()['pred_masks'],
                    mask_file
                )
                gc.collect()  # Force collect because GPU memory errors are common.
            del predictor
            gc.collect()  # Force collect because GPU memory errors are common.

    @staticmethod
    def __merge_instances(instances: torch.Tensor, mask_file: str):
        """
        Merges instances into a mask file. If the mask file already exists, it is merged.
        """

        # Note: The instance masks are torch bool
        image_mask_0 = torch.zeros(instances.shape[1:3], dtype=torch.bool, device=instances.device)
        # The loop ends up being more memory efficient than summing across the third axis.
        for i in range(instances.shape[0]):
            image_mask_0 = image_mask_0 | instances[i]
        image_mask_0 = image_mask_0.cpu().numpy()

        if os.path.isfile(mask_file):
            image_mask_1: np.ndarray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            image_mask_1 = image_mask_1.astype(np.bool)
            image_mask_0 = image_mask_0 | image_mask_1

        cv2.imwrite(mask_file, image_mask_0.astype(np.uint8), [cv2.IMWRITE_PNG_BILEVEL, 1])


class ResinPredictor:
    """
    Similar to detectron2 DefaultPredictor, but allows for multiple image transformations.
    """

    def __init__(self, cfg: CfgNode):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        check_pointer = DetectionCheckpointer(self.model)
        check_pointer.load(cfg.MODEL.WEIGHTS)

        self.tfm_gens = [EqualizeTransformGen()]

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    @torch.no_grad()
    def __call__(self, image):
        if self.input_format == "RGB":
            image = image[:, :, ::-1]
        image, transforms = apply_transform_gens(self.tfm_gens, image)
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return self.model([inputs])[0]
