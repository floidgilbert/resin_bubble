import argparse
import json
import os
import warnings
from typing import List

import torch
import torch.multiprocessing as mp
from detectron2.config import CfgNode, get_cfg

from resin_bubble.data import list_sort_images
from resin_bubble.predict import EnsembleBatchPredictor, get_masks_info
from resin_bubble.visualize import create_video_from_stills, plot_predictions, visualize_masks


def __get_cla_parser() -> argparse.ArgumentParser:
    """
    Command line parser.
    """

    description = """Batch prediction. Ensembles (multiple models) are supported. Separate instances within and across 
multiple models are discriminated using intersection over union (IOU)."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--input',
        '-i',
        metavar='INPUT_DIR',
        help='Directory containing images.',
        required=True,
    )
    parser.add_argument(
        '--output',
        '-o',
        metavar='OUTPUT_DIR',
        help='Directory to receive predictions.',
        required=True,
    )
    parser.add_argument(
        '--device',
        '-d',
        metavar='cuda | cpu',
        choices=('cuda', 'cpu'),
        help='Device for predictions.',
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--num_gpu',
        metavar='NUM_GPU',
        help='Number of GPUs to use. By default, all GPUs are used.',
        type=int,
        default=torch.cuda.device_count(),
    )
    parser.add_argument(
        '--num_cpu',
        metavar='NUM_CPU',
        help='Number of CPUs to use. By default, all CPUs are used.',
        type=int,
        default=mp.cpu_count(),
    )
    parser.add_argument(
        '--config',
        '-cfg',
        metavar='CONFIG_FILE',
        help='YAML file containing base model configuration.',
        default='./configs/resin_bubble.yaml',
    )
    parser.add_argument(
        '--models',
        '-m',
        nargs='+',
        metavar='MODEL',
        help='One or more PyTorch model files to use for prediction.',
        default=['./models/default.pth']
    )
    parser.add_argument(
        '--min_area',
        metavar='MIN_AREA',
        default=10,
        type=int,
        help='Minimum instance area in pixels. Default 10.',
    )
    parser.add_argument(
        '--num_images',
        metavar='NUM',
        type=int,
        help='Number of image files in INPUT_DIR to process.',
    )
    return parser


def __predict_masks(
        process_index: int,
        process_count: int,
        input_files: List[str],
        mask_directory: str,
        cfg: CfgNode,
        models: List[str],
        use_cuda: bool,
):
    """
    Launch prediction task. Designed for parallel processing via MP spawn.
    """

    # TODO: Select idle GPUs only.
    if use_cuda:
        torch.cuda.set_device(process_index)
        print(f'Using GPU {process_index}')
    predictor = EnsembleBatchPredictor(cfg, models)
    chunk_size = len(input_files) // process_count
    which_files = slice(
        process_index * chunk_size,
        (process_index + 1) * chunk_size if (process_index + 1) < process_count else None
    )
    predictor(input_files[which_files], mask_directory)


def __print_summary(info: dict):
    """
    Print summary of prediction results.
    """

    instances_retained = 0
    instances_filtered = 0
    for d in info.values():
        instances_retained += len(d['instances'])
        instances_filtered += len(d['filtered_by_min_area'])

    print(f'Images: {len(info)}')
    print(f'Instances retained: {instances_retained}')
    print(f'Instances filtered: {instances_filtered}')


if __name__ == '__main__':
    args = __get_cla_parser().parse_args()
    print('Command Line: ', args, '\n')

    # Load configuration
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    if args.device:
        cfg.MODEL.DEVICE = args.device

    # Select files
    input_files = list_sort_images(args.input)
    if len(input_files) == 0:
        raise ValueError(f'The input folder does not contain valid images. ({args.input})')
    if args.num_images:
        input_files = input_files[0:args.num_images]

    # Predict and create masks in parallel
    mask_directory = os.path.join(args.output, 'mask')
    process_count = 0
    use_cuda = False
    if cfg.MODEL.DEVICE == 'cuda':
        if (not torch.cuda.is_available()) | (torch.cuda.device_count() == 0):
            raise RuntimeError('Device \'cuda\' is not available. Use device \'cpu\'.')
        if args.num_gpu == 0:
            raise RuntimeError('Device \'cuda\' is specified but requested GPUs is zero. Use device \'cpu\'.')
        process_count = min(torch.cuda.device_count(), args.num_gpu, mp.cpu_count(), len(input_files))
        if process_count < args.num_gpu:
            warnings.warn(f'Using {process_count} GPUs when {args.num_gpu} were requested.')
        use_cuda = True
    else:
        process_count = min(args.num_cpu, mp.cpu_count(), len(input_files))
        if process_count < args.num_cpu:
            warnings.warn(f'Using {process_count} CPUs when {args.num_cpu} were requested.')

    if process_count == 1:
        __predict_masks(0, 1, input_files, mask_directory, cfg, args.models, use_cuda)
    else:
        mp.spawn(
            __predict_masks,
            (process_count, input_files, mask_directory, cfg, args.models, use_cuda),
            process_count
        )

    # Summarize mask information
    mask_info = get_masks_info(mask_directory, args.min_area)
    mask_info_file = os.path.join(args.output, 'info.json')
    with open(mask_info_file, 'w') as f:
        json.dump(mask_info, f)

    # Generate visualizations.
    create_video_from_stills(mask_directory, os.path.join(args.output, 'mask.avi'))

    viz_directory = os.path.join(args.output, 'viz')
    visualize_masks(mask_directory, args.input, viz_directory)
    create_video_from_stills(viz_directory, os.path.join(args.output, 'viz.avi'))

    viz_rect_directory = os.path.join(args.output, 'viz-rect')
    visualize_masks(mask_directory, args.input, viz_rect_directory, mask_info, 'VizRects')
    create_video_from_stills(viz_rect_directory, os.path.join(args.output, 'viz-rect.avi'))

    # Generate plots
    plot_predictions(mask_info, args.output)

    # Print summary
    print()
    __print_summary(mask_info)
