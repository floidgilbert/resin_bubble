import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer

from .data import ResinDataMapper
from .predict import ResinPredictor

__all__ = ['ResinTrainer', 'save_model_state_dict']


def save_model_state_dict(config_file: str, model_file: str, output_file: str):
    """
    Save PyTorch model state_dict to file for distribution.
    """

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_file
    predictor = ResinPredictor(cfg)
    torch.save(predictor.model.state_dict(), output_file)


class ResinTrainer(DefaultTrainer):
    """
    Superclass detectron2 DefaultTrainer to use ResinDataMapper, which adds custom image transformations.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, ResinDataMapper(cfg, True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        raise NotImplementedError
