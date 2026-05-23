"""
Visualize detections. Can be used to visually evaluate detector performance without tracking component.
"""
import argparse
import logging
import os
import statistics
from typing import Dict, List, Optional

from tqdm import tqdm
import yaml

from motrack.datasets import dataset_factory
from motrack.utils.logging import configure_logging

logger = logging.getLogger('Tool-VisualizeDetections')


def load_yaml_params(file_path: Optional[str]) -> Dict:
    """
    Load parameters from a YAML file if the path is provided and valid.

    Args:
        file_path: Path to the YAML file.

    Returns:
        The loaded parameters from the YAML file, or an empty dictionary.
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description='Dataset loader script.')
    parser.add_argument('--dataset-type', required=True, help='Type of the dataset (e.g., COCO, ImageNet).')
    parser.add_argument('--dataset-path', required=True, help='Path to the dataset.')
    parser.add_argument('--dataset-params-path', help='Path to the dataset params YAML file.', default=None)
    parser.add_argument(
        '--image-resolution',
        type=int,
        nargs=2,
        help='Input image resolution (width height).',
        default=[1500, 800]
    )
    parser.add_argument(
        '--crop-resolution',
        type=int,
        nargs=2,
        help='Input crop resolution (width height).',
        default=[224, 112]
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Runs dataset pixel density analysis.

    Args:
        args: Parsed command line arguments.

    Ignore Returns: This function returns None.
    """
    image_pixel_densities: List[float] = []
    crop_pixel_densities: List[float] = []

    image_heights: List[int] = []
    image_widths: List[int] = []
    crop_heights: List[int] = []
    crop_widths: List[int] = []

    for split in tqdm(['train', 'val', 'test'], desc='Analysing splits', unit='split', leave=False):
        dataset = dataset_factory(
            dataset_type=args.dataset_type,
            path=os.path.join(args.dataset_path, split),
            params=load_yaml_params(args.dataset_params_path),
            test=split == 'test'
        )

        for scene_name in tqdm(dataset.scenes, desc='Analysing scenes', unit='split', leave=False):
            scene_info = dataset.get_scene_info(scene_name)
            image_pixel_density = min(1.0, args.image_resolution[0] / scene_info.imwidth) \
                * min(1.0, args.image_resolution[1] / scene_info.imheight)
            image_pixel_densities.append(image_pixel_density)
            image_heights.append(scene_info.imheight)
            image_widths.append(scene_info.imwidth)

            object_ids = dataset.get_scene_object_ids(scene_name)

            for index in range(scene_info.seqlength):
                for object_id in object_ids:
                    data = dataset.get_object_data_by_frame_index(object_id, index)
                    if data is None:
                        continue

                    max_w = int(data.bbox[2] * scene_info.imwidth)
                    max_h = int(data.bbox[3] * scene_info.imheight)
                    crop_pixel_density = min(1.0, args.crop_resolution[0] / max_w) \
                        * min(1.0, args.crop_resolution[1] / max_h)
                    crop_pixel_densities.append(crop_pixel_density)
                    crop_heights.append(max_h)
                    crop_widths.append(max_w)

    avg_image_pixel_density = statistics.mean(image_pixel_densities)
    stddev_image_pixel_density = statistics.stdev(image_pixel_densities)
    avg_crop_pixel_density = statistics.mean(crop_pixel_densities)
    stddev_crop_pixel_density = statistics.stdev(crop_pixel_densities)

    avg_image_height = statistics.mean(image_heights)
    avg_image_width = statistics.mean(image_widths)
    avg_crop_height = statistics.mean(crop_heights)
    avg_crop_width = statistics.mean(crop_widths)

    messages = [
        'Pixel density stats:',
        f'\tImage mean pixel density: {avg_image_pixel_density:.2f}',
        f'\tImage sttdev pixel density: {stddev_image_pixel_density:.2f}',
        f'\tCrop mean pixel density: {avg_crop_pixel_density:.2f}',
        f'\tCrop sttdev pixel density: {stddev_crop_pixel_density:.2f}',
        'Heights/Width stats:',
        f'\tImage mean height: {avg_image_height:.2f}',
        f'\tImage mean width: {avg_image_width:.2f}',
        f'\tCrop mean height: {avg_crop_height:.2f}',
        f'\tCrop mean width: {avg_crop_width:.2f}'
    ]

    logger.info('\n'.join(messages))


if __name__ == '__main__':
    configure_logging()
    main(parse_args())
