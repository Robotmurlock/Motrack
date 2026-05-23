"""Hydra CLI wrapper around ``motrack.tools.optimization.run_optimize``."""
import hydra

from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.tools.optimization import run_optimize
from motrack.utils import pipeline


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='optimization/tpe_sort', version_base='1.1')
@pipeline.task('optimize')
def main(cfg: GlobalConfig) -> None:
    run_optimize(cfg)


if __name__ == '__main__':
    main()
