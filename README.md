# MA-WAM release

[Paper](https://arxiv.org/abs/2609.31281) · [Project page](https://ma-wam.github.io/) · [Models](https://huggingface.co/ma-wam/MA-WAM) · [Datasets](https://huggingface.co/datasets/Guowei-Zou/CoFlow-datasets)

By [Guowei Zou](https://guowei-zou.github.io/Guowei-Zou/) and collaborators.

This repository contains the source code for **MA-WAM (Multi-Agent World-Action Model)**, a test-time planner for frozen multi-agent flow policies.  At each environment step, the planner samples candidate joint action sequences from a CoFlow policy, rolls them forward with a routed world model, scores their predicted team returns, and executes the first action of the highest-scoring sequence.

The release combines the CoFlow policy/evaluator implementation with the MA-WAM world model, planner, scorer baselines, and paper-oriented validation scripts.  It is a source-only snapshot created on 2026-07-28.

## What is included

- `diffuser/`: CoFlow policy, dataset interfaces, environments, and evaluator code.
- `world_model/`: the routed dynamics/reward model, rollout code, and scorer variants.
- `run_scripts/`: world-model training, test-time planning, same-state ranking, timing, fidelity, and scorer-ablation entry points.
- `exp_specs/`: CoFlow policy and MA-WAM world-model configurations for MPE, MAMuJoCo, and SMAC.
- `scripts/`: data conversion and utility scripts.

The following assets are deliberately excluded: offline datasets, policy checkpoints, world-model checkpoints, W&B files, experiment logs, and generated results.  The source workspaces contained about 63 GB of datasets alone, so including them would make the code release impractical and would not clarify their redistribution status.

## Environment

The snapshot was exercised on Python 3.8.20, PyTorch 1.12.1+cu113, CUDA 11.3, NumPy 1.24.4, and PyYAML 6.0.  Create the tested base environment and install this package from the repository root:

```bash
conda env create -f environment.yml
conda activate ma-wam
pip install -e .
```

`requirements.txt` contains the Python packages used by the source code.  MPE, MAMuJoCo, and SMAC also require their benchmark-specific simulator assets.  In particular, install the compatible MPE and MAMuJoCo packages before running those domains, install PySC2/SMAC and the StarCraft II game assets before running SMAC, and configure MuJoCo for MAMuJoCo.  These benchmark components are not vendored here.

## Data and checkpoint layout

The release preserves the data paths used by the original experiments:

```text
MA-WAM-release/
├── data/mpe/<task>/<quality>/
│   ├── obs.npy
│   ├── actions.npy
│   ├── rewards.npy
│   └── path_lengths.npy
└── diffuser/datasets/data/
    ├── mamujoco/<task>/<quality>/
    └── smac/<task>/<quality>/
```

For SMAC, a dataset directory also needs `legals.npy`.  The planner receives the policy run directory and the MA-WAM checkpoint explicitly, so they may live outside this repository.  A policy run directory must retain its evaluator configuration and checkpoint subtree, for example `.../run/100/checkpoint/state_*.pt`.

### Public dataset sources and preparation

MA-WAM uses the same public datasets as the CoFlow backbone experiments.

- **MPE (Spread, Tag, and World).** Download the original OMAR data release
  from [the OMAR repository](https://github.com/ling-pan/OMAR):
  [Cooperative Navigation / Spread](https://drive.google.com/file/d/1YVk_ajtvbcq8R2m0u0RasfB0csToV7XP/view?usp=sharing),
  [Predator-Prey / Tag](https://pan.baidu.com/s/16W-UyyCtfKDt9oTgeNOhJA)
  (extraction code: `m7vw`), and
  [World](https://pan.baidu.com/s/1pjZmeIAlaepPpug3b5olGA)
  (extraction code: `5k3t`). Download the `expert`, `medium-replay`,
  `medium`, and `random` splits, and place their per-agent seed directories
  under `diffuser/datasets/data/mpe/<task>/<quality>/`.

  The frozen policy reads this per-agent layout. The MA-WAM world model uses a
  flattened version of the same trajectories. Create it with:

  ```bash
  python scripts/convert_mpe_data.py \
    --envs simple_spread simple_tag simple_world \
    --src-root diffuser/datasets/data/mpe \
    --dst-root data/mpe
  ```

- **MA-MuJoCo and SMAC.** Download the `2xAnt` and `4xAnt` MA-MuJoCo splits,
  and the SMAC `3m`, `8m`, `2s3z`, and `5m_vs_6m` splits, from the
  [official OG-MARL dataset collection](https://huggingface.co/datasets/InstaDeepAI/og-marl).
  Use the `Good`, `Medium`, and `Poor` qualities reported in the paper. Export
  the datasets into the NumPy layout shown above; MA-MuJoCo needs `obs.npy`,
  `actions.npy`, `rewards.npy`, and `path_lengths.npy`, while SMAC additionally
  needs `legals.npy`. Install the matching MuJoCo or StarCraft II/SMAC assets
  before environment evaluation.

The source release excludes datasets and learned parameters. Public resources are linked above; use checkpoints and configurations matching the evaluated setting.

## Core commands

Train an MA-WAM world model with one of the supplied configurations:

```bash
python run_scripts/train_world_model.py \
  --config exp_specs/phase1_wm/mpe/simple_spread/wm_spread_medium.yaml \
  --gpu 0
```

## Policy-training seeds

Every CoFlow policy configuration under `exp_specs/` defines the same five training seeds:

```yaml
variables:
  seed: [100, 200, 300, 400, 500]
```

`run_experiment.py` expands this list into five policy-training variants.  The supplied configurations keep `meta_data.num_workers: 1`, so a launcher invocation runs those variants sequentially unless that worker count is increased deliberately for the available hardware.

Evaluate receding-horizon planning for a continuous-control policy:

```bash
python run_scripts/plan_eval.py \
  --run_dir /path/to/policy/run/100 \
  --wm /path/to/best_model.pt \
  --n_agents 3 --obs_dim 18 --act_dim 2 \
  --M 8 --H 8 --num_eval 16 --steps 5 --gpu 0
```

For a discrete SMAC task, replace `--act_dim` with `--num_actions <N> --discrete`.  `--ablate_random` runs the matched random-selector control in the same invocation.  The paired evaluator (`run_scripts/plan_eval_paired.py`) is the appropriate entry point when the two selection arms must share stochastic inputs.

The remaining paper-facing entry points are grouped by purpose:

| Purpose | Entry points |
| --- | --- |
| Scorer architecture and selector controls | `train_baseline_full_wm.py`, `train_monolithic_full.py`, `train_routing_variant_wm.py`, `plan_eval_scorer.py`, `plan_eval_matched_scorer.py`, `plan_eval_q_ranker.py`, `plan_eval_return_predictor.py`, `plan_eval_routing_variant.py` |
| Ranking and fidelity checks | `eval_same_state_ranking.py`, `eval_arch_same_state_influence.py`, `validate_wm_fidelity.py`, `validate_routing_counterfactual.py` |
| Runtime and figures | `profile_plan_timing_all.py`, `plot_paper1_planning_visuals.py` |

Each evaluator exposes its required arguments through `--help`.  Do not substitute a checkpoint trained with a different number of agents, observation size, action representation, or routing architecture.

## Source-only verification

After installing the environment, this command checks every packaged Python module without starting a training run or requiring datasets:

```bash
python -m compileall -q diffuser world_model run_scripts scripts run_experiment.py
```

## License and upstream component

The CoFlow base implementation included in this release retains its MIT license in `LICENSE`.  MA-WAM extends that implementation with its routed world model and test-time planner.

## Independent training seeds

Policy configurations use five training seeds: `100, 200, 300, 400, 500`.
The auxiliary training entry points now use the same five seeds by default,
run sequentially in separate processes. Initialization and training sampling
use the selected seed; each run saves under `seed_<seed>/` in its output
directory, including a `training_seed.json` record.

Use `--training-seeds 100 200 300` for three runs,
`--training-seeds 100` for a single run, or `--dry-run` to inspect the
commands without loading data or training. Evaluation seeds and dataset
source seeds are separate. These defaults configure new training runs;
they do not establish that historical results or released checkpoints
contain five independently trained models.

This applies to `run_scripts/train_world_model.py`, `train_monolithic_full.py`, `train_baseline_full_wm.py`, `train_routing_variant_wm.py`, and `train_trajectory_return_predictor.py`. Use the chosen seed subdirectory when supplying a world-model checkpoint to an evaluator. The return predictor retains its separate `--split_seed` for the episode split.

## Benchmark dependencies

Use a separate environment for each repository: the projects share the
`diffuser` package name and must not be installed together. The default
`requirements.txt` supports the MPE training and evaluation path. Legacy
Gym requires the pip/setuptools/wheel bootstrap versions shown above.
The previous all-in-one dependency list is retained as
`requirements-historical.txt` for reference, not as the installation command.

For SMAC, additionally install `requirements-smac.txt` and StarCraft II
with the appropriate maps. For MA-MuJoCo, install `requirements-mujoco.txt`,
MuJoCo 2.1.0, and set `LD_LIBRARY_PATH` to include its `bin` directory.
D4RL/mjrl and TensorFlow dataset converters are optional legacy integrations,
not required to train from the supplied MPE NumPy layout.

Verify the installation before providing datasets:

```bash
python scripts/check_install.py
```

On a minimal Linux host, install a C/C++ compiler, Python development headers,
libcurl/OpenSSL development headers (for the logger's pycurl dependency),
and OpenGL runtime libraries before pip installation. Headless runs can set
`SDL_AUDIODRIVER=dummy`.

The supported examples use vector observations in MPE, SMAC, and MA-MuJoCo.
Inherited image-policy and PyBullet prototypes are not part of the tested
release workflow. Full benchmark training and all historical checkpoints
are not certified by the short installation/runtime checks.

See [release verification](VERIFICATION.md) for the tested installation and runtime paths and their scope.

MPE Tag and World also require the frozen opponent file `pretrained_adv_model.pt` under `diffuser/datasets/data/mpe/simple_tag/` or `simple_world/`, respectively. Spread does not require this opponent asset.
