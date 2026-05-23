"""Hydra CLI wrapper around ``motrack.tools.inference.run_inference``."""
import hydra

from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.tools.inference import run_inference
from motrack.utils import pipeline


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
@pipeline.task('inference')
def main(cfg: GlobalConfig) -> None:
    run_inference(cfg)


if __name__ == '__main__':
    main()
