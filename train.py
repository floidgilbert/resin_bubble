import argparse
import os

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_setup, launch

from resin_bubble.data import load_annotations_from_masks
from resin_bubble.train import ResinTrainer


def __get_cla_parser():

    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument(
        '--input',
        '-i',
        metavar='INPUT_DIR',
        help='Directory containing images.',
        required=True,
    )
    parser.add_argument(
        '--masks',
        '-m',
        metavar='MASK_DIR',
        help='Directory containing masks.',
        required=True,
    )
    parser.add_argument(
        '--output',
        '-o',
        metavar='OUTPUT_DIR',
        help='Directory to receive model and other output.',
        required=True,
    )
    parser.add_argument(
        '--iterations',
        '-iter',
        metavar='ITER',
        help='Number of training iterations.',
        type=int,
    )
    parser.add_argument(
        '--learning_rate',
        '-lr',
        metavar='RATE',
        help='Learning rate to train the model.',
        type=float,
    )
    parser.add_argument(
        '--checkpoint_period',
        '-chk',
        metavar='PERIOD',
        help='Number of training iterations for each checkpoint.',
        type=int,
    )
    parser.add_argument(
        '--num_gpu',
        '-g',
        metavar='NUM_GPU',
        help='Number of GPUs to use. By default, all GPUs are used.',
        type=int,
        default=torch.cuda.device_count(),
    )
    parser.add_argument(
        '--config',
        '-c',
        metavar='CFG_FILE',
        help='YAML file containing base model configuration.',
        default='./configs/resin_bubble.yaml',
    )
    parser.add_argument(
        '--weights',
        '-w',
        metavar='WEIGHTS_FILE',
        help='PyTorch model or weights file to start from. By default, downloads a model pre-trained on COCO.',
    )
    parser.add_argument(
        '--resume',
        '-r',
        action='store_true',
        help='Resume from last available checkpoint.',
    )
    parser.add_argument(
        '--local_rank',
        help='Used internally for detectron2 PyTorch launcher.',
        type=int,
        default=0,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line.',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args: argparse.Namespace):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output
    if args.iterations:
        cfg.SOLVER.MAX_ITER = args.iterations
    if args.learning_rate:
        cfg.SOLVER.BASE_LR = args.learning_rate
    if args.checkpoint_period:
        cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    # See https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
    cfg.SOLVER.IMS_PER_BATCH *= args.num_gpu
    cfg.freeze()
    default_setup(cfg, args)

    _ = load_annotations_from_masks(args.input, args.masks)

    def get_train_annotations():
        return [_[i] for i in range(len(_))]

    DatasetCatalog.register(cfg.DATASETS.TRAIN[0], get_train_annotations)
    meta_train = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    meta_train.set(thing_classes=['bubble'])

    # data.visualize_metadata(DATASETS.TRAIN[0], '../data/resin_bubble/viz')

    # def get_test_annotations():
    #     return [_[i] for i in [2, 5, 6]]

    # DatasetCatalog.register(DATASETS.TEST[0], get_test_annotations)
    # meta_test = MetadataCatalog.get(DATASETS.TEST[0])
    # meta_test.set(thing_classes=['bubble'])

    trainer = ResinTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = __get_cla_parser().parse_args()
    print('Command Line: ', args, '\n')
    os.makedirs(args.output, exist_ok=True)
    launch(main, args.num_gpu, args=(args,))
