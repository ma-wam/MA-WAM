import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_scripts.plan_eval import (  # noqa: E402
    find_best_step,
    load_policy_ckpt,
    load_worker,
    load_wm,
    set_seed,
    to_np,
    wm_score,
)


TASKS = []
for env in ["simple_spread", "simple_tag", "simple_world"]:
    for split in ["expert", "medium", "medium-replay", "random"]:
        TASKS.append(("MPE", env, split))
for env in ["2ant", "4ant"]:
    for split in ["Good", "Medium", "Poor"]:
        TASKS.append(("MAMuJoCo", env, split))
for env in ["3m", "2s3z", "5m_vs_6m", "8m"]:
    for split in ["Good", "Medium", "Poor"]:
        TASKS.append(("SMAC", env, split))


def data_path(domain, env, split):
    if domain == "MPE":
        return os.path.join(PROJECT_ROOT, "data", "mpe", env, split)
    env_type = {"MAMuJoCo": "mamujoco", "SMAC": "smac"}[domain]
    return os.path.join(
        PROJECT_ROOT,
        "diffuser",
        "datasets",
        "data",
        env_type,
        env,
        split,
    )


def infer_dims(domain, env, split):
    root = data_path(domain, env, split)
    obs = np.load(os.path.join(root, "obs.npy"), mmap_mode="r")
    acts = np.load(os.path.join(root, "actions.npy"), mmap_mode="r")
    n_agents = int(obs.shape[-2])
    obs_dim = int(obs.shape[-1])
    if domain == "SMAC":
        legals = os.path.join(root, "legals.npy")
        if os.path.exists(legals):
            num_actions = int(np.load(legals, mmap_mode="r").shape[-1])
        else:
            num_actions = int(np.max(acts[: min(len(acts), 10000)])) + 1
        return n_agents, obs_dim, num_actions, True
    return n_agents, obs_dim, int(acts.shape[-1]), False


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def run_dir_for(domain, env, split):
    if domain == "MPE":
        return first_match(os.path.join(
            PROJECT_ROOT,
            "logs",
            "aug_m2flow_mpe",
            f"{env}-{split}",
            "*",
            "100",
        ))
    if domain == "MAMuJoCo":
        return os.path.join(
            PROJECT_ROOT,
            "logs",
            "base_coflowc_mamujoco",
            f"{env}-{split}",
            "run",
            "100",
        )
    return first_match(os.path.join(
        PROJECT_ROOT,
        "logs",
        "aug_m2flow_smac",
        f"{env}-{split}",
        "*",
        "100",
    ))


def wm_path(env, split):
    return os.path.join(PROJECT_ROOT, "logs", "phase1_wm", env, split, "best_model.pt")


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate_candidates(worker, obs_queue, returns, env_ts, norm, model, M, H, discrete, num_actions, legal):
    C = worker.Config
    A = C.n_agents
    od = norm.observation_dim
    first_acts = []
    wm_acts = []
    obs_in = np.stack(list(obs_queue), axis=1)
    for _ in range(M):
        samples = worker._generate_samples(obs_in, returns, env_ts)
        Hm = min(H, samples.shape[1] - 1)
        seq_logits = []
        for tt in range(Hm):
            oc = torch.cat([samples[:, tt], samples[:, tt + 1]], dim=-1).reshape(-1, A, 2 * od)
            if C.share_inv or C.joint_inv:
                if C.joint_inv:
                    a = model.inv_model(oc.reshape(oc.shape[0], -1)).reshape(oc.shape[0], A, -1)
                else:
                    a = model.inv_model(oc)
            else:
                a = torch.stack([model.inv_model[i](oc[:, i]) for i in range(A)], dim=1)
            seq_logits.append(to_np(a))
        seq = np.stack(seq_logits, axis=1)

        if discrete:
            idx = seq.argmax(axis=-1)
            masked0 = seq[:, 0].copy()
            masked0[legal.astype(int) == 0] = -1e9
            idx[:, 0] = masked0.argmax(axis=-1)
            onehot = np.zeros((*idx.shape, num_actions), dtype=np.float32)
            np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
            wm_acts.append(onehot)
            first_acts.append(idx[:, 0])
        else:
            raw = norm.unnormalize(seq, "actions")
            wm_acts.append(raw)
            first_acts.append(raw[:, 0])
    return np.stack(first_acts, axis=0), np.stack(wm_acts, axis=0)


@torch.no_grad()
def profile_one(args, domain, env, split):
    run_dir = run_dir_for(domain, env, split)
    ckpt = wm_path(env, split)
    if not run_dir or not os.path.isdir(run_dir):
        return {"status": "missing_run_dir", "domain": domain, "task": env, "quality": split, "run_dir": run_dir}
    if not os.path.exists(ckpt):
        return {"status": "missing_wm", "domain": domain, "task": env, "quality": split, "wm": ckpt}

    n_agents, obs_dim, act_dim, discrete = infer_dims(domain, env, split)
    num_actions = act_dim if discrete else None

    set_seed(args.seed)
    worker = load_worker(run_dir)
    device = worker.Config.device
    load_step = find_best_step(run_dir)
    load_policy_ckpt(worker, load_step)
    worker.Config.num_eval = args.num_envs
    worker.Config.num_envs = min(args.num_envs, worker.Config.num_envs)
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    wm = load_wm(
        ckpt,
        n_agents,
        obs_dim,
        act_dim,
        device,
        args.n_dyn,
        args.n_rew,
        args.hidden,
        args.n_slots,
    )

    C = worker.Config
    norm = worker.normalizer
    model = worker.trainer.ema_model
    E = min(args.num_envs, len(worker.env_list), C.num_envs)
    envs = worker.env_list[:E]
    returns = (C.test_ret * torch.ones(E, 1, C.n_agents)).to(device)
    env_ts = (torch.arange(C.horizon + C.history_horizon) - C.history_horizon).to(device)
    env_ts = env_ts.unsqueeze(0).expand(E, -1)
    obs = np.concatenate([e.reset()[None] for e in envs], axis=0)
    obs_queue = deque(maxlen=C.history_horizon + 1)
    if getattr(C, "use_zero_padding", False):
        obs_queue.extend([np.zeros_like(obs) for _ in range(C.history_horizon)])
    else:
        obs_queue.extend([norm.normalize(obs, "observations") for _ in range(C.history_horizon)])

    m1_gen, m8_gen, wm_times = [], [], []
    total_steps = args.warmup + args.profile_steps
    for step in range(total_steps):
        raw_obs = obs.copy()
        obs_queue.append(norm.normalize(obs, "observations"))
        legal = None
        if discrete:
            legal = np.stack([e.get_legal_actions() for e in envs], axis=0)

        sync_if_cuda()
        t0 = time.perf_counter()
        first1, _ = generate_candidates(worker, obs_queue, returns, env_ts, norm, model, 1, args.H, discrete, num_actions, legal)
        sync_if_cuda()
        t1 = time.perf_counter()

        sync_if_cuda()
        t2 = time.perf_counter()
        first8, acts8 = generate_candidates(worker, obs_queue, returns, env_ts, norm, model, args.M, args.H, discrete, num_actions, legal)
        sync_if_cuda()
        t3 = time.perf_counter()

        sync_if_cuda()
        t4 = time.perf_counter()
        scores = to_np(wm_score(wm, raw_obs, acts8, device))
        sync_if_cuda()
        t5 = time.perf_counter()

        if step >= args.warmup:
            m1_gen.append((t1 - t0) * 1000.0)
            m8_gen.append((t3 - t2) * 1000.0)
            wm_times.append((t5 - t4) * 1000.0)

        best = scores.argmax(axis=0)
        chosen = np.stack([first8[best[i], i] for i in range(E)], axis=0)
        new_obs = []
        for i, e in enumerate(envs):
            try:
                o, r, d, info = e.step(chosen[i])
                if np.asarray(d).all():
                    o = e.reset()
                new_obs.append(o[None])
            except Exception:
                new_obs.append(obs[i][None])
        obs = np.concatenate(new_obs, axis=0)
        env_ts = env_ts + 1

    def mean_std(xs):
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    m1_mean, m1_std = mean_std(m1_gen)
    m8_mean, m8_std = mean_std(m8_gen)
    wm_mean, wm_std = mean_std(wm_times)
    return {
        "status": "ok",
        "domain": domain,
        "task": env,
        "quality": split,
        "num_envs": E,
        "profile_steps": args.profile_steps,
        "warmup": args.warmup,
        "M": args.M,
        "H": args.H,
        "policy_steps": args.steps,
        "m1_gen_ms": m1_mean,
        "m1_gen_std_ms": m1_std,
        "m8_gen_ms": m8_mean,
        "m8_gen_std_ms": m8_std,
        "wm_score_ms": wm_mean,
        "wm_score_std_ms": wm_std,
        "wm_share_pct": 100.0 * wm_mean / max(m8_mean + wm_mean, 1e-9),
        "run_dir": run_dir,
        "wm": ckpt,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--out_dir", default="results/paper1_timing_all")
    p.add_argument("--domains", default="MPE,MAMuJoCo,SMAC")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--profile_steps", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_dyn", type=int, default=8)
    p.add_argument("--n_rew", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_slots", type=int, default=4)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.makedirs(args.out_dir, exist_ok=True)
    domains = {x.strip() for x in args.domains.split(",") if x.strip()}
    records = []
    for domain, env, split in TASKS:
        if domain not in domains:
            continue
        print(f"[PROFILE] {domain} {env}-{split}", flush=True)
        try:
            rec = profile_one(args, domain, env, split)
        except Exception as exc:
            rec = {"status": "error", "domain": domain, "task": env, "quality": split, "error": repr(exc)}
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    json_path = os.path.join(args.out_dir, "paper1_timing_all.json")
    csv_path = os.path.join(args.out_dir, "paper1_timing_all.csv")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    keys = [
        "status", "domain", "task", "quality", "num_envs", "profile_steps", "M", "H",
        "m1_gen_ms", "m1_gen_std_ms", "m8_gen_ms", "m8_gen_std_ms",
        "wm_score_ms", "wm_score_std_ms", "wm_share_pct", "run_dir", "wm", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rec in records:
            w.writerow({k: rec.get(k, "") for k in keys})
    print(f"[SAVED] {json_path}")
    print(f"[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
