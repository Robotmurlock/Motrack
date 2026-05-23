"""Hydra CLI wrapper around ``motrack.tools.eval.run_eval``."""
import hydra

from motrack.common.project import DANCETRACK_TRACKERS_CONFIG_PATH
from motrack.config_parser import GlobalConfig
from motrack.tools.eval import run_eval
from motrack.utils import pipeline


@hydra.main(config_path=DANCETRACK_TRACKERS_CONFIG_PATH, config_name='movesort', version_base='1.1')
@pipeline.task('eval')
def main(cfg: GlobalConfig) -> None:
    run_eval(cfg)

    from motrack.tools.mlflow_logger import load_and_log_run
    load_and_log_run(cfg)


if __name__ == '__main__':
    main()
