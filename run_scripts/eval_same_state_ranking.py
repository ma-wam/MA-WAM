import argparse
import copy
import csv
import json
import os
import sys
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
from run_scripts.profile_plan_timing_all import infer_dims, run_dir_for, wm_path  # noqa: E402


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corrcoef(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pairwise_acc(true_scores, pred_scores):
    total = 0
    correct = 0
    m = len(true_scores)
    for i in range(m):
        for j in range(i + 1, m):
            dt = true_scores[i] - true_scores[j]
            dp = pred_scores[i] - pred_scores[j]
            if abs(dt) < 1e-12:
                continue
            total += 1
            correct += int(dt * dp > 0)
    return float(correct / total) if total else float("nan")


@torch.no_grad()
def generate_candidates(worker, obs_queue, returns, env_ts, M, H, discrete, num_actions):
    config = worker.Config
    norm = worker.normalizer
    model = worker.trainer.ema_model
    n_agents = config.n_agents
    obs_dim = norm.observation_dim
    obs_in = np.stack(list(obs_queue), axis=1)
    first_acts = []
    wm_acts = []
    exec_acts = []

    legal = None
    if discrete:
        legal = np.stack([worker.env_list[0].get_legal_actions()], axis=0)

    for _ in range(M):
        samples = worker._generate_samples(obs_in, returns, env_ts)
        horizon = min(H, samples.shape[1] - 1)
        seq_logits = []
        for tt in range(horizon):
            obs_cat = torch.cat([samples[:, tt], samples[:, tt + 1]], dim=-1).reshape(
                -1, n_agents, 2 * obs_dim
            )
            if config.share_inv or config.joint_inv:
                if config.joint_inv:
                    act = model.inv_model(obs_cat.reshape(obs_cat.shape[0], -1)).reshape(
                        obs_cat.shape[0], n_agents, -1
                    )
                else:
                    act = model.inv_model(obs_cat)
            else:
                act = torch.stack([model.inv_model[i](obs_cat[:, i]) for i in range(n_agents)], dim=1)
            seq_logits.append(to_np(act))
        seq = np.stack(seq_logits, axis=1)

        if discrete:
            idx = seq.argmax(axis=-1)
            masked0 = seq[:, 0].copy()
            masked0[legal.astype(int) == 0] = -1e9
            idx[:, 0] = masked0.argmax(axis=-1)
            onehot = np.zeros((*idx.shape, num_actions), dtype=np.float32)
            np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
            wm_acts.append(onehot)
            exec_acts.append(idx)
            first_acts.append(idx[:, 0])
        else:
            raw = norm.unnormalize(seq, "actions")
            wm_acts.append(raw)
            exec_acts.append(raw)
            first_acts.append(raw[:, 0])

    return np.stack(first_acts, axis=0), np.stack(wm_acts, axis=0), np.stack(exec_acts, axis=0)


def base_env(env):
    cur = env
    while hasattr(cur, "env") and not hasattr(cur, "world"):
        cur = cur.env
    return cur


def wrapper_state_chain(env):
    chain = []
    cur = env
    while hasattr(cur, "env"):
        chain.append(cur)
        cur = cur.env
    return chain


def snapshot_mpe_env(env):
    base = base_env(env)
    world = base.world
    entities = list(world.agents) + list(world.landmarks)
    entity_state = []
    for ent in entities:
        rec = {
            "p_pos": np.array(ent.state.p_pos, copy=True),
            "p_vel": np.array(ent.state.p_vel, copy=True),
        }
        if hasattr(ent.state, "c"):
            rec["c"] = np.array(ent.state.c, copy=True)
        if hasattr(ent, "action"):
            rec["action_u"] = np.array(getattr(ent.action, "u", np.zeros(0)), copy=True)
            rec["action_c"] = np.array(getattr(ent.action, "c", np.zeros(0)), copy=True)
        entity_state.append(rec)
    snap = {
        "time": getattr(base, "time", None),
        "entities": entity_state,
        "wrappers": [],
    }
    for wrapper in wrapper_state_chain(env):
        rec = {}
        if hasattr(wrapper, "prey_obs") and wrapper.prey_obs is not None:
            rec["prey_obs"] = np.array(wrapper.prey_obs, copy=True)
        snap["wrappers"].append(rec)
    for name in ["cache_dists", "cached_dist_vect", "cached_dist_mag"]:
        if hasattr(world, name):
            value = getattr(world, name)
            snap[name] = None if value is None else np.array(value, copy=True)
    return snap


def restore_mpe_env(env, snap):
    base = base_env(env)
    world = base.world
    if snap["time"] is not None:
        base.time = snap["time"]
    entities = list(world.agents) + list(world.landmarks)
    for ent, rec in zip(entities, snap["entities"]):
        ent.state.p_pos = np.array(rec["p_pos"], copy=True)
        ent.state.p_vel = np.array(rec["p_vel"], copy=True)
        if "c" in rec:
            ent.state.c = np.array(rec["c"], copy=True)
        if hasattr(ent, "action") and "action_u" in rec:
            ent.action.u = np.array(rec["action_u"], copy=True)
            ent.action.c = np.array(rec["action_c"], copy=True)
    for name in ["cache_dists", "cached_dist_vect", "cached_dist_mag"]:
        if name in snap:
            setattr(world, name, None if snap[name] is None else np.array(snap[name], copy=True))
    for wrapper, rec in zip(wrapper_state_chain(env), snap.get("wrappers", [])):
        if "prey_obs" in rec:
            wrapper.prey_obs = np.array(rec["prey_obs"], copy=True)


# --- MAMuJoCo exact state snapshot/restore -----------------------------------
# Unlike MPE we cannot rely on copy.deepcopy(env): mujoco_py's MjSim/MjData are
# Cython objects that do not deepcopy reliably. Instead we use the canonical
# mujoco_py state API (sim.get_state()/set_state()), which fully captures the
# physics state (qpos, qvel, act, time). We also restore the episode-step
# counters in the wrapper chain (MujocoMulti.steps, TimeLimit._elapsed_steps) so
# that done/truncation behaves identically for every candidate from the same
# anchor state.
_MUJOCO_COUNTER_ATTRS = ("steps", "_elapsed_steps")


def _env_chain_generic(env):
    chain = []
    seen = set()
    stack = [env]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        chain.append(cur)
        for attr in ("env", "wrapped_env", "timelimit_env", "_env"):
            child = getattr(cur, attr, None)
            if child is None or id(child) in seen:
                continue
            if hasattr(child, "step") or hasattr(child, "sim"):
                stack.append(child)
    return chain


def find_mujoco_sim(env):
    for cur in _env_chain_generic(env):
        sim = getattr(cur, "sim", None)
        if sim is not None and hasattr(sim, "get_state") and hasattr(sim, "set_state"):
            return sim
    return None


def snapshot_mujoco_env(env):
    sim = find_mujoco_sim(env)
    if sim is None:
        raise RuntimeError(
            "Could not locate a mujoco_py sim in the env wrapper chain; "
            "MAMuJoCo same-state restore requires sim.get_state()/set_state()."
        )
    st = sim.get_state()
    snap = {
        "state_cls": type(st),
        "time": float(st.time),
        "qpos": np.array(st.qpos, copy=True),
        "qvel": np.array(st.qvel, copy=True),
        "act": None if st.act is None else np.array(st.act, copy=True),
        "udd_state": copy.deepcopy(st.udd_state) if st.udd_state else {},
        "counters": [],
    }
    for cur in _env_chain_generic(env):
        for attr in _MUJOCO_COUNTER_ATTRS:
            if attr in vars(cur):
                snap["counters"].append((cur, attr, vars(cur)[attr]))
    return snap


def restore_mujoco_env(env, snap):
    sim = find_mujoco_sim(env)
    st = snap["state_cls"](
        snap["time"],
        np.array(snap["qpos"], copy=True),
        np.array(snap["qvel"], copy=True),
        None if snap["act"] is None else np.array(snap["act"], copy=True),
        copy.deepcopy(snap["udd_state"]),
    )
    sim.set_state(st)
    sim.forward()
    for cur, attr, value in snap["counters"]:
        setattr(cur, attr, value)


def rollout_candidate(env, actions, discrete, snapshot=None, restore_fn=None):
    if snapshot is None or restore_fn is None:
        env_copy = copy.deepcopy(env)
    else:
        env_copy = env
        restore_fn(env_copy, snapshot)
    total = 0.0
    done = False
    for act in actions:
        if done:
            break
        action = np.asarray(act)
        if discrete:
            action = action.astype(np.int64)
        obs, reward, done_flag, info = env_copy.step(action)
        total += float(np.asarray(reward, dtype=np.float64).reshape(-1).sum())
        done = bool(np.asarray(done_flag).all())
    if snapshot is not None and restore_fn is not None:
        restore_fn(env_copy, snapshot)
    return total


def evaluate(args):
    run_dir = args.run_dir or run_dir_for(args.domain, args.task, args.quality)
    wm_ckpt = args.wm or wm_path(args.task, args.quality)
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Missing run_dir for {args.domain} {args.task}-{args.quality}: {run_dir}")
    if not os.path.exists(wm_ckpt):
        raise FileNotFoundError(f"Missing world model checkpoint: {wm_ckpt}")

    n_agents, obs_dim, act_dim, discrete = infer_dims(args.domain, args.task, args.quality)
    num_actions = act_dim if discrete else None

    set_seed(args.seed)
    worker = load_worker(run_dir)
    device = worker.Config.device
    load_step = args.load_step if args.load_step > 0 else find_best_step(run_dir)
    load_policy_ckpt(worker, load_step)
    worker.Config.num_eval = 1
    worker.Config.num_envs = 1
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    wm = load_wm(wm_ckpt, n_agents, obs_dim, act_dim, device, args.n_dyn, args.n_rew, args.hidden, args.n_slots)

    config = worker.Config
    norm = worker.normalizer
    env = worker.env_list[0]
    returns = (config.test_ret * torch.ones(1, 1, config.n_agents)).to(device)
    env_ts = (torch.arange(config.horizon + config.history_horizon) - config.history_horizon).to(device)
    env_ts = env_ts.unsqueeze(0)
    obs = env.reset()[None]
    obs_queue = deque(maxlen=config.history_horizon + 1)
    if getattr(config, "use_zero_padding", False):
        obs_queue.extend([np.zeros_like(obs) for _ in range(config.history_horizon)])
    else:
        obs_queue.extend([norm.normalize(obs, "observations") for _ in range(config.history_horizon)])

    if args.domain == "MPE":
        snapshot_fn, restore_fn = snapshot_mpe_env, restore_mpe_env
    elif args.domain == "MAMuJoCo":
        snapshot_fn, restore_fn = snapshot_mujoco_env, restore_mujoco_env
    else:
        # SMAC (StarCraft II engine) has no reliable mid-episode state restore;
        # fall back to copy.deepcopy(env) inside rollout_candidate.
        snapshot_fn, restore_fn = None, None
    selfcheck_done = False

    rows = []
    episode_t = 0
    for state_id in range(args.num_states):
        obs_queue.append(norm.normalize(obs, "observations"))
        raw_obs = obs.copy()
        first_acts, wm_acts, exec_acts = generate_candidates(
            worker, obs_queue, returns, env_ts, args.M, args.H, discrete, num_actions
        )
        pred = to_np(wm_score(wm, raw_obs, wm_acts, device)).reshape(args.M)
        snapshot = snapshot_fn(env) if snapshot_fn is not None else None
        true = np.array(
            [rollout_candidate(env, exec_acts[m, 0], discrete, snapshot, restore_fn) for m in range(args.M)],
            dtype=np.float64,
        )
        if snapshot is not None and not selfcheck_done:
            # One-time determinism check: re-running candidate 0 from the restored
            # anchor state must reproduce true[0] exactly. If state restore were
            # broken, the second rollout would start from a drifted state.
            recheck = rollout_candidate(env, exec_acts[0, 0], discrete, snapshot, restore_fn)
            if abs(recheck - true[0]) > 1e-5:
                raise RuntimeError(
                    f"[SELFCHECK FAIL] {args.domain} {args.task}-{args.quality}: state restore "
                    f"non-deterministic (true[0]={true[0]:.8f}, recheck={recheck:.8f})"
                )
            print(
                f"[SELFCHECK PASS] {args.domain} {args.task}-{args.quality}: state restore "
                f"deterministic (true[0]={true[0]:.8f}, recheck={recheck:.8f})",
                flush=True,
            )
            selfcheck_done = True
        pred_best = int(np.argmax(pred))
        true_best = int(np.argmax(true))
        true_worst = int(np.argmin(true))
        oracle_gap = float(true[true_best] - true[true_worst])
        regret = float(true[true_best] - true[pred_best])
        random_regret = float(true[true_best] - np.mean(true))
        top2 = set(np.argsort(pred)[-2:].tolist())
        rows.append(
            {
                "domain": args.domain,
                "task": args.task,
                "quality": args.quality,
                "state_id": state_id,
                "M": args.M,
                "H": args.H,
                "pred_best": pred_best,
                "true_best": true_best,
                "top1": int(pred_best == true_best),
                "top2": int(true_best in top2),
                "spearman": corrcoef(rankdata(true), rankdata(pred)),
                "pairwise_acc": pairwise_acc(true, pred),
                "regret": regret,
                "normalized_regret": regret / max(oracle_gap, 1e-8),
                "random_regret": random_regret,
                "oracle_return": float(true[true_best]),
                "selected_return": float(true[pred_best]),
                "random_mean_return": float(np.mean(true)),
                "oracle_gap": oracle_gap,
            }
        )

        chosen = first_acts[pred_best, 0]
        if snapshot is not None:
            restore_fn(env, snapshot)
        next_obs, reward, done, info = env.step(chosen.astype(np.int64) if discrete else chosen)
        if getattr(config, "use_return_to_go", False):
            returns[0] = worker._update_return_to_go(returns[0], reward)
        episode_t += 1
        if bool(np.asarray(done).all()) or episode_t >= config.max_path_length - 1:
            obs = env.reset()[None]
            returns = (config.test_ret * torch.ones(1, 1, config.n_agents)).to(device)
            env_ts = (torch.arange(config.horizon + config.history_horizon) - config.history_horizon).to(device)
            env_ts = env_ts.unsqueeze(0)
            obs_queue.clear()
            if getattr(config, "use_zero_padding", False):
                obs_queue.extend([np.zeros_like(obs) for _ in range(config.history_horizon)])
            else:
                obs_queue.extend([norm.normalize(obs, "observations") for _ in range(config.history_horizon)])
            episode_t = 0
        else:
            obs = next_obs[None]
            env_ts = env_ts + 1

    return rows


def summarize(rows):
    keys = ["top1", "top2", "spearman", "pairwise_acc", "regret", "normalized_regret", "random_regret", "oracle_gap"]
    out = {"n_states": len(rows)}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.nanmean(vals))
    out["selected_vs_random_gain"] = float(np.nanmean([r["selected_return"] - r["random_mean_return"] for r in rows]))
    out["oracle_vs_selected_gap"] = float(np.nanmean([r["oracle_return"] - r["selected_return"] for r in rows]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="MPE", choices=["MPE", "MAMuJoCo", "SMAC"])
    parser.add_argument("--task", default="simple_spread")
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--wm", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_states", type=int, default=64)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--out_dir", default="results/same_state_ranking")
    parser.add_argument("--n_dyn", type=int, default=8)
    parser.add_argument("--n_rew", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_slots", type=int, default=4)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = evaluate(args)
    summary = summarize(rows)
    stem = f"{args.domain}_{args.task}_{args.quality}_M{args.M}_H{args.H}".replace("/", "_")
    csv_path = os.path.join(args.out_dir, f"{stem}.csv")
    json_path = os.path.join(args.out_dir, f"{stem}_summary.json")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "rows": rows[:5]}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {json_path}")


if __name__ == "__main__":
    main()
