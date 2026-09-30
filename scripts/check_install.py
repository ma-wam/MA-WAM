"""Dataset-free installation check for the supported MPE path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import diffuser.utils
from diffuser.datasets.mpe import load_environment


def main():
    env = load_environment("simple_spread-medium")
    try:
        obs = np.asarray(env.reset())
        assert np.isfinite(obs).all()
        action = [np.zeros(space.shape, dtype=np.float32) for space in env.action_space]
        obs, reward, done, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
    finally:
        env.close()
    print("MPE reset/step and policy imports passed; torch=" + torch.__version__)


if __name__ == "__main__":
    main()
