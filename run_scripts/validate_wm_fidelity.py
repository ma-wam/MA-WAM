"""Paired WM-vs-simulator fidelity validation for MAMuJoCo.
From each live-env anchor (exact sim snapshot), apply the SAME action sequence to
(a) the world model (autoregressive rollout) and (b) the real simulator (replay),
then measure per-step observation divergence. Answers: are WM-imagined trajectories
dynamically plausible, and up to what horizon can we trust them?
"""
import argparse, os, sys, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_scripts.plan_eval import load_worker, load_wm
from run_scripts.profile_plan_timing_all import infer_dims, run_dir_for, wm_path, data_path
from run_scripts.eval_same_state_ranking import (
    snapshot_mujoco_env, restore_mujoco_env,
    snapshot_mpe_env, restore_mpe_env,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="MAMuJoCo", choices=["MAMuJoCo", "MPE"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--n-anchors", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--act-noise", type=float, default=0.1, help="generation action noise std (matches gen)")
    ap.add_argument("--act-std", type=float, default=0.3, help="fallback random action std if no dataset actions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    domain = args.domain
    snapshot_fn = snapshot_mujoco_env if domain == "MAMuJoCo" else snapshot_mpe_env
    restore_fn = restore_mujoco_env if domain == "MAMuJoCo" else restore_mpe_env
    n_agents, obs_dim, act_dim, discrete = infer_dims(domain, args.task, args.quality)
    run_dir = run_dir_for(domain, args.task, args.quality)
    wm_ckpt = wm_path(args.task, args.quality)

    worker = load_worker(run_dir)
    device = worker.Config.device
    env = worker.env_list[0]
    wm = load_wm(wm_ckpt, n_agents, obs_dim, act_dim, device)

    # per-(agent,dim) obs std from dataset, for scale-normalized divergence
    obs_file = os.path.join(data_path(domain, args.task, args.quality), "obs.npy")
    real_obs = np.load(obs_file, mmap_mode="r")
    samp = np.asarray(real_obs[: min(200000, len(real_obs))], dtype=np.float64)
    obs_std = samp.reshape(-1, n_agents, obs_dim).std(axis=0)  # (A, od)
    # active dims = meaningful variance; near-constant dims (std~0) carry no dynamics info
    active = obs_std > 0.05 * np.median(obs_std)
    obs_std_safe = np.where(active, obs_std, 1.0)
    n_active = int(active.sum())

    # real behavior action windows (faithful to generation: dataset actions + small noise)
    act_file = os.path.join(data_path(domain, args.task, args.quality), "actions.npy")
    real_acts = np.load(act_file, mmap_mode="r")  # (Nsteps, A, ad) flat
    n_steps_total = len(real_acts)
    _asamp = np.asarray(real_acts[: min(100000, n_steps_total)], dtype=np.float32)
    act_lo, act_hi = float(_asamp.min()), float(_asamp.max())

    H = args.horizon
    max_path = getattr(worker.Config, "max_path_length", 1000)

    # accumulators: per-step squared error (raw + normalized), averaged over anchors
    sq_raw = np.zeros(H)      # mean over active (agent,dim) of (wm-sim)^2, per step
    sq_norm = np.zeros(H)     # same but each active dim divided by its std (z-score)
    sq_persist = np.zeros(H)  # persistence baseline: (anchor - sim)^2 normalized, active dims
    counts = np.zeros(H)
    selfcheck = None
    used = 0

    obs = env.reset()
    ep_t = 0
    for a in range(args.n_anchors):
        # diversify anchor: take a few random steps; reset if near episode end
        if ep_t >= max_path - H - 2:
            obs = env.reset(); ep_t = 0
        k = rng.randint(0, 25)
        for _ in range(k):
            ra = np.clip(rng.normal(0, args.act_std, size=(n_agents, act_dim)), -1, 1)
            obs, _, done, _ = env.step(ra); ep_t += 1
            if np.asarray(done).all() or ep_t >= max_path - H - 2:
                obs = env.reset(); ep_t = 0
        anchor_obs = np.asarray(obs, dtype=np.float32).reshape(n_agents, obs_dim)

        # faithful action sequence: a real dataset action window + generation noise (std 0.1)
        s0 = rng.randint(0, n_steps_total - H)
        base = np.asarray(real_acts[s0:s0 + H], dtype=np.float32).reshape(H, n_agents, act_dim)
        act_seq = (base + rng.normal(0, args.act_noise, size=base.shape)).astype(np.float32)
        act_seq = np.clip(act_seq, act_lo, act_hi)

        snap = snapshot_fn(env)

        # (b) simulator replay
        sim_obs = np.zeros((H, n_agents, obs_dim), dtype=np.float64)
        restore_fn(env, snap)
        cur = anchor_obs.copy()
        broke = False
        for t in range(H):
            o, _, done, _ = env.step(act_seq[t])
            sim_obs[t] = np.asarray(o, dtype=np.float64).reshape(n_agents, obs_dim)
            if np.asarray(done).all():
                broke = True; sim_obs = sim_obs[: t + 1]; break

        # determinism self-check on first anchor: replay again must match
        if selfcheck is None:
            restore_fn(env, snap)
            o2 = env.step(act_seq[0])[0]
            selfcheck = float(np.abs(np.asarray(o2).reshape(n_agents, obs_dim) - sim_obs[0]).max())

        Huse = sim_obs.shape[0]

        # (a) WM autoregressive rollout from the same anchor obs
        with torch.no_grad():
            cur_t = torch.tensor(anchor_obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,A,od)
            wm_obs = np.zeros((Huse, n_agents, obs_dim), dtype=np.float64)
            for t in range(Huse):
                at = torch.tensor(act_seq[t], dtype=torch.float32, device=device).unsqueeze(0)  # (1,A,ad)
                nxt = wm.predict_next_obs(cur_t, at)
                wm_obs[t] = nxt.squeeze(0).cpu().numpy()
                cur_t = nxt

        # restore env to anchor so the next outer step continues cleanly
        restore_fn(env, snap)

        diff = wm_obs - sim_obs                  # (Huse, A, od)  WM vs sim
        dpers = anchor_obs[None] - sim_obs        # (Huse, A, od)  persistence (no-change) vs sim
        am = active[None]                          # broadcast active mask
        nA = max(1, n_active)
        zdiff = (diff / obs_std_safe[None]) * am
        zpers = (dpers / obs_std_safe[None]) * am
        sq_raw[:Huse] += (diff ** 2 * am).sum(axis=(1, 2)) / nA
        sq_norm[:Huse] += (zdiff ** 2).sum(axis=(1, 2)) / nA
        sq_persist[:Huse] += (zpers ** 2).sum(axis=(1, 2)) / nA
        counts[:Huse] += 1
        used += 1
        # advance the outer episode by one real step (use first action) to keep moving
        obs, _, done, _ = env.step(act_seq[0]); ep_t += 1
        if np.asarray(done).all():
            obs = env.reset(); ep_t = 0

    rmse_raw = np.sqrt(sq_raw / np.maximum(counts, 1))
    rmse_norm = np.sqrt(sq_norm / np.maximum(counts, 1))
    rmse_persist = np.sqrt(sq_persist / np.maximum(counts, 1))
    out = {
        "task": args.task, "quality": args.quality, "n_anchors": used,
        "horizon": H, "selfcheck_max_abs": selfcheck, "n_active_dims": n_active,
        "counts": counts.tolist(),
        "rmse_raw_per_step": rmse_raw.tolist(),
        "rmse_norm_per_step": rmse_norm.tolist(),
        "rmse_persistence_per_step": rmse_persist.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[DONE] {args.task}-{args.quality} anchors={used} active_dims={n_active} "
          f"selfcheck={selfcheck:.1e} | WM z-RMSE s1={rmse_norm[0]:.2f} s{H//2}={rmse_norm[H//2]:.2f} "
          f"s{H}={rmse_norm[-1]:.2f} | persist s{H}={rmse_persist[-1]:.2f}")


if __name__ == "__main__":
    main()
