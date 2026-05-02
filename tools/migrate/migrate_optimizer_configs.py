"""
One-shot migration: rewrite legacy ``optimizer.mfgcs:`` blocks in saved
config snapshots into ``optimizer.sampler_params:``.

Studies completed before the sampler-instantiation refactor have config
snapshots like::

    optimizer:
      sampler: mfgcs
      mfgcs: {scene_sampler: {...}, coordinate_optimizer: {...}, ...}
      sampler_params: {}

This script rewrites them in place to the new shape::

    optimizer:
      sampler: mfgcs
      sampler_params: {scene_sampler: {...}, coordinate_optimizer: {...}, ...}

Optuna snapshots (``sampler != 'mfgcs'``) are left untouched.

Usage::

    python -m tools.migrate.migrate_optimizer_configs <root_dir> [--dry-run]

``<root_dir>`` is typically the project's ``path.master``; the script
recursively walks for ``run_configs/<hash>/config.yaml`` files and rewrites
each one. Idempotent — already-migrated snapshots are no-ops.

Tagging in MLflow is not attempted here: MLflow params are immutable, and
re-uploading per-run config artifacts requires the original tracking URI
and run IDs that this offline script does not see. The loader-side
fallback in ``tools/analysis/report_optimization.py`` handles un-migrated
*MLflow* runs transparently; this script is for the on-disk snapshots
only.
"""
import argparse
import logging
import os
import sys
from typing import List

import yaml

from motrack.common import conventions

logger = logging.getLogger('migrate-optimizer-configs')


def _migrate_optimizer_block(opt: dict) -> bool:
    """Rewrite a single ``optimizer:`` dict in place. Returns True on change."""
    if opt.get('sampler') != 'mfgcs':
        return False
    legacy = opt.get('mfgcs')
    if not legacy:
        return False
    sampler_params = opt.get('sampler_params') or {}
    if sampler_params:
        # Already-migrated snapshots have a populated sampler_params; the
        # legacy ``mfgcs`` key is leftover and just needs to be removed.
        opt.pop('mfgcs', None)
        return True
    opt['sampler_params'] = dict(legacy)
    opt.pop('mfgcs', None)
    return True


def _migrate_snapshot(path: str, *, dry_run: bool) -> bool:
    """Migrate a single ``config.yaml`` snapshot. Returns True on change."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return False
    opt = cfg.get('optimizer')
    if not isinstance(opt, dict):
        return False
    if not _migrate_optimizer_block(opt):
        return False
    if dry_run:
        logger.info(f'[dry-run] would migrate {path}')
        return True
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f'migrated {path}')
    return True


def _find_snapshots(root: str) -> List[str]:
    """Walk ``root`` for ``run_configs/<hash>/config.yaml`` snapshots."""
    matches: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == conventions.RUN_CONFIGS_DIRNAME:
            for hash_dir in dirnames:
                cfg_path = os.path.join(
                    dirpath, hash_dir, conventions.CONFIG_FILENAME,
                )
                if os.path.exists(cfg_path):
                    matches.append(cfg_path)
    return matches


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='Root directory (typically path.master).')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report what would change without writing files.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Verbose logging.',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(name)s %(levelname)s: %(message)s',
    )

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        logger.error(f'root directory not found: {root}')
        return 2

    snapshots = _find_snapshots(root)
    logger.info(f'scanning {len(snapshots)} config snapshots under {root}')

    changed = 0
    for path in snapshots:
        try:
            if _migrate_snapshot(path, dry_run=args.dry_run):
                changed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(f'failed to migrate {path}: {exc}')

    verb = 'would migrate' if args.dry_run else 'migrated'
    logger.info(f'{verb} {changed}/{len(snapshots)} snapshots')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
