"""Independent training runs; evaluation and data-source seeds are separate."""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

DEFAULT_SEEDS = [100, 200, 300, 400, 500]


def add_training_arguments(parser):
    parser.add_argument('--training-seeds', type=int, nargs='+', default=None,
                        help='Independent training seeds (default: 100 200 300 400 500)')
    parser.add_argument('--dry-run', action='store_true', help='Print training commands only')


def prepare_training(args, legacy_seed=None):
    explicit = getattr(args, legacy_seed, None) if legacy_seed else None
    if explicit is not None and args.training_seeds is not None:
        raise ValueError('Use either the single-seed option or --training-seeds')
    seeds = args.training_seeds if args.training_seeds is not None else ([explicit] if explicit is not None else DEFAULT_SEEDS)
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Training seeds must be unique integers in [0, 2**32)')
    if len(seeds) > 1 or args.dry_run:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument('--training-seeds', nargs='+', type=int)
        parser.add_argument('--dry-run', action='store_true')
        if legacy_seed:
            parser.add_argument('--' + legacy_seed, type=int)
        _, rest = parser.parse_known_args(sys.argv[1:])
        for seed in seeds:
            command = [sys.executable, str(Path(sys.argv[0]).resolve()), *rest,
                       '--training-seeds', str(seed)]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True, env=dict(os.environ, PYTHONHASHSEED=str(seed)))
        return True
    args.training_seed = seeds[0]
    if legacy_seed:
        setattr(args, legacy_seed, seeds[0])
    if hasattr(args, 'gpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np
    import torch
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)
    return False


def seed_output(path, args):
    path = Path(path) / ('seed_' + str(args.training_seed))
    path.mkdir(parents=True, exist_ok=True)
    (path / 'training_seed.json').write_text(json.dumps({'training_seed': args.training_seed}) + '\n')
    return str(path)
