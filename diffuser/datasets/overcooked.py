"""
Overcooked environment + offline dataset adapter for the CoFlow / CoFlow-WM
diffuser framework.  Mirrors diffuser/datasets/smac_env.py.

Key detail: the underlying OverCookedEnv default observation is the lossless
8056-dim encoding, but the offline datasets (and the Phase-1 world model) are
built on the 96-dim featurized state produced by
``mdp.featurize_state(state, mlam)`` (exactly as in overcooked_gen.py).
This wrapper therefore overrides the observation to that 96-dim featurization
so that eval-time observations match the training distribution.
"""
import os
import sys
from typing import Any, Dict, List, Optional

import gym
import numpy as np

# Install overcooked_bic, or point this variable to its source checkout.
_OC_ROOT = os.environ.get("MA_WAM_OVERCOOKED_ROOT")
if _OC_ROOT and _OC_ROOT not in sys.path:
    sys.path.insert(0, _OC_ROOT)

N_ACTIONS = 6  # Overcooked joint action set per agent (Action.NUM_ACTIONS)


class Overcooked(gym.Env):
    """Multi-agent Overcooked wrapper matching the SMAC-style interface."""

    metadata = {}

    def __init__(self, layout: str, episode_length: int = 400):
        from overcooked_ai_py.env import OverCookedEnv

        self._environment = OverCookedEnv(scenario=layout, episode_length=episode_length)
        self._oc = self._environment.overcooked
        self._mdp = self._oc.mdp
        self._mlam = getattr(self._oc, "mlam", None)
        self.num_agents = 2
        self.num_actions = N_ACTIONS
        self.max_episode_length = episode_length
        self._agents = [f"agent_{n}" for n in range(self.num_agents)]
        self._done = False

        obs0 = self._featurize()
        self.obs_dim = obs0.shape[-1]  # 96

        self.observation_space = [
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            gym.spaces.Discrete(n=self.num_actions) for _ in range(self.num_agents)
        ]

    def _featurize(self) -> np.ndarray:
        """96-dim per-agent featurization matching the offline data."""
        feat = self._mdp.featurize_state(self._oc.state, self._mlam)
        return np.asarray(feat, dtype=np.float32)  # (num_agents, 96)

    def reset(self):
        self._environment.reset()
        self._done = False
        return self._featurize()

    def step(self, actions: np.ndarray):
        actions = [int(a) for a in np.asarray(actions).reshape(-1)[: self.num_agents]]
        # OverCookedEnv.step applies _convert_action (int->Action) then advances state;
        # we ignore its lossless obs and featurize ourselves below.
        _, reward, done, info = self._environment.step(actions)
        self._done = bool(done)
        # per-agent reward: use shaped + sparse (matches overcooked_gen.py)
        shaped = np.asarray(info.get("shaped_r_by_agent", [0] * self.num_agents), dtype=np.float32)
        sparse = np.asarray(info.get("sparse_r_by_agent", [0] * self.num_agents), dtype=np.float32)
        reward_n = (shaped + sparse).astype(np.float32)
        if reward_n.shape[0] != self.num_agents:
            reward_n = np.array([float(reward)] * self.num_agents, dtype=np.float32)
        done_n = np.array([self._done] * self.num_agents)
        return self._featurize(), reward_n, done_n, info

    def env_done(self) -> bool:
        return self._done

    def get_legal_actions(self) -> np.ndarray:
        """All 6 actions are always available in Overcooked."""
        return np.ones((self.num_agents, self.num_actions), dtype="float32")

    def get_stats(self) -> Optional[Dict]:
        return {}

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def possible_agents(self) -> List:
        return self._agents

    @property
    def environment(self):
        return self._environment

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        return getattr(self.__dict__["_environment"], name)


def load_environment(name, **kwargs):
    if type(name) is not str:
        return name

    idx = name.find("-")
    env_name, data_split = name[:idx], name[idx + 1:]

    env = Overcooked(env_name, **kwargs)
    if not hasattr(env, "metadata") or not isinstance(env.metadata, dict):
        env.metadata = {}
    env.metadata["data_split"] = data_split
    env.metadata["name"] = env_name
    env.metadata["global_feats"] = ["states"]
    return env


def sequence_dataset(env, preprocess_fn):
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "data/overcooked",
        env.metadata["name"],
        env.metadata["data_split"],
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory not found: {}".format(dataset_path))

    observations = np.load(os.path.join(dataset_path, "obs.npy"))      # (T, n_agents, 96)
    rewards = np.load(os.path.join(dataset_path, "rewards.npy"))       # (T, n_agents)
    actions = np.load(os.path.join(dataset_path, "actions.npy"))       # (T, n_agents) discrete idx
    path_lengths = np.load(os.path.join(dataset_path, "path_lengths.npy"))

    n_agents = observations.shape[1]
    # all actions always legal in Overcooked
    legal_actions = np.ones((observations.shape[0], n_agents, N_ACTIONS), dtype=np.float32)

    start = 0
    for path_length in path_lengths:
        path_length = int(path_length)
        end = start + path_length
        episode_data = {}
        episode_data["observations"] = observations[start:end]
        episode_data["legal_actions"] = legal_actions[start:end]
        episode_data["rewards"] = rewards[start:end]
        episode_data["actions"] = actions[start:end]
        episode_data["terminals"] = np.zeros((path_length, n_agents), dtype=bool)
        episode_data["terminals"][-1] = True
        yield episode_data
        start = end
