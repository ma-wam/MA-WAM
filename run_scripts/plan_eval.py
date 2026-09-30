"""
Test-time planning evaluation (WM-as-protagonist pilot).

At each real env step: sample M candidate action sequences from the trained policy
(diffusion is stochastic), roll each through the MoE world model for H steps, score by
predicted cumulative reward, execute the FIRST action of the best candidate
(receding-horizon MPC). Reuses an EXISTING policy run_dir + Phase1 WM checkpoint.
No retraining.

  M=1  -> policy-only (should reproduce training-time eval baseline)  [harness check]
  M>1  -> WM planning. If reward(M>1) > reward(M=1), the WM adds test-time value.

Usage (SMAC 3m-Good on gzdj):
  python run_scripts/plan_eval.py -g 0 \
    --run_dir logs/aug_coflowc_smac_v6_r10/3m-Good/<run>/100 \
    --wm logs/phase1_wm/3m/Good/best_model.pt \
    --n_agents 3 --obs_dim 33 --num_actions 9 --discrete \
    --M 8 --H 10 --num_eval 16 --steps 5
"""
import os
import sys
import argparse
import csv
import time
import numpy as np
import torch
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ml_logger parses process arguments during the evaluator import.  Temporarily
# defer this script's help flag so argparse can render this entry point's help.
_deferred_help_args = [arg for arg in ("-h", "--help") if arg in sys.argv]
for _arg in _deferred_help_args:
    sys.argv.remove(_arg)
import diffuser.datasets
from diffuser.datasets.augmented_sequence import AugmentedSequenceDataset
diffuser.datasets.AugmentedSequenceDataset = AugmentedSequenceDataset
from diffuser.utils.evaluator import MADEvaluatorWorker
from world_model import MoEWorldModel
sys.argv.extend(_deferred_help_args)


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_worker(run_dir):
    w = MADEvaluatorWorker.__new__(MADEvaluatorWorker)
    w.verbose = False
    w.initialized = False
    w._init(log_dir=run_dir)
    return w


def find_best_step(run_dir):
    csv_path = os.path.join(run_dir, "results", "evaluation_history.csv")
    if os.path.exists(csv_path):
        best_step, best = None, -1e18
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    s = int(row["step"]); rw = float(row["average_ep_reward_mean"])
                except (ValueError, KeyError):
                    continue
                if rw > best:
                    best, best_step = rw, s
        return best_step
    else:
        # Fallback: use latest checkpoint when no evaluation_history.csv
        import glob
        ckpts = glob.glob(os.path.join(run_dir, "checkpoint", "state_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {run_dir}/checkpoint/")
        steps = [int(os.path.basename(c).replace("state_", "").replace(".pt", "")) for c in ckpts]
        steps = [s for s in steps if s > 0]
        best_step = max(steps)
        print(f"No evaluation_history.csv found; using latest checkpoint: state_{best_step}.pt")
        return best_step


def load_policy_ckpt(worker, load_step):
    """Replicate MADEvaluatorWorker._evaluate's checkpoint load (model + ema)."""
    C = worker.Config
    lp = os.path.join(worker.log_dir, "checkpoint", f"state_{load_step}.pt")
    sd = torch.load(lp, map_location=C.device)
    sd["model"] = {k: v for k, v in sd["model"].items() if "value_diffusion_model." not in k}
    sd["ema"] = {k: v for k, v in sd["ema"].items() if "value_diffusion_model." not in k}
    worker.trainer.step = sd["step"]
    worker.trainer.model.load_state_dict(sd["model"])
    worker.trainer.ema_model.load_state_dict(sd["ema"])
    print(f"Loaded policy checkpoint state_{load_step}.pt")


def load_wm(ckpt, n_agents, obs_dim, act_dim, device, n_dyn=8, n_rew=4, hidden=256, n_slots=4):
    wm = MoEWorldModel(
        n_agents=n_agents, obs_dim=obs_dim, act_dim=act_dim,
        n_dyn_experts=n_dyn, n_rew_experts=n_rew, hidden_dim=hidden, n_slots=n_slots,
    )
    ck = torch.load(ckpt, map_location=device)
    wm.load_state_dict(ck["model_state_dict"])
    return wm.to(device).eval()


@torch.no_grad()
def wm_score(wm, start_obs_raw, act_seqs, device):
    """start_obs_raw: (E, A, od) raw env obs.
       act_seqs: (M, E, H, A, ad) WM-space actions (one-hot for discrete, raw for cont).
       returns: (M, E) cumulative predicted reward."""
    M, E, H, A, ad = act_seqs.shape
    obs = torch.as_tensor(start_obs_raw, dtype=torch.float32, device=device)
    obs = obs.unsqueeze(0).expand(M, E, A, -1).reshape(M * E, A, -1).contiguous()
    acts = torch.as_tensor(act_seqs, dtype=torch.float32, device=device).reshape(M * E, H, A, ad)
    total = torch.zeros(M * E, device=device)
    for t in range(H):
        a = acts[:, t]
        nxt = wm.predict_next_obs(obs, a)
        r = wm.predict_reward(obs, a, nxt)          # (M*E, A, 1)
        total = total + r.reshape(M * E, -1).sum(dim=1)
        obs = nxt
    return total.reshape(M, E)


@torch.no_grad()
def planning_eval(worker, wm, num_episodes, M, H, device, num_actions=None, discrete=False, select="wm", obs_noise_std=0.0):
    C = worker.Config
    norm = worker.normalizer
    envs = worker.env_list[:num_episodes]
    A = C.n_agents
    od = norm.observation_dim
    model = worker.trainer.ema_model
    # dedicated RNG for random-selection ablation (keeps env/global RNG paired across arms)
    sel_rng = np.random.RandomState(getattr(C, "seed", 0) + 12345)
    cuda = torch.cuda.is_available()
    t_gen, t_score, nstep = 0.0, 0.0, 0

    dones = [0] * num_episodes
    ep_rew = [np.zeros(A) for _ in range(num_episodes)]
    wins = np.zeros(num_episodes)

    returns = (C.test_ret * torch.ones(num_episodes, 1, A)).to(device)
    env_ts = (torch.arange(C.horizon + C.history_horizon) - C.history_horizon).to(device)
    env_ts = env_ts.unsqueeze(0).expand(num_episodes, -1)

    obs = np.concatenate([e.reset()[None] for e in envs], axis=0)  # (E, A, od) raw
    if obs_noise_std > 0:
        obs = obs + np.random.randn(*obs.shape).astype(obs.dtype) * obs_noise_std
    obs_queue = deque(maxlen=C.history_horizon + 1)
    if getattr(C, "use_zero_padding", False):
        obs_queue.extend([np.zeros_like(obs) for _ in range(C.history_horizon)])
    else:
        obs_queue.extend([norm.normalize(obs, "observations") for _ in range(C.history_horizon)])

    t = 0
    while sum(dones) < num_episodes:
        raw_obs = obs.copy()                                   # for WM rollout (raw space)
        nobs = norm.normalize(obs, "observations")
        obs_queue.append(nobs)
        obs_in = np.stack(list(obs_queue), axis=1)             # (E, hist+1, A, od)

        legal = None
        if discrete:
            legal = np.stack([e.get_legal_actions() for e in envs], axis=0)  # (E, A, num_actions)

        if cuda:
            torch.cuda.synchronize()
        _tg = time.perf_counter()
        first_acts = []     # executable first action per candidate: (E, A) idx  or (E,A,ad)
        wm_acts = []        # WM-space action seq per candidate: (E, Hm, A, ad)
        for m in range(M):
            samples = worker._generate_samples(obs_in, returns, env_ts)   # (E, horizon, A, od) normed
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
                seq_logits.append(to_np(a))                    # (E, A, ad/num_actions)
            seq = np.stack(seq_logits, axis=1)                 # (E, Hm, A, *)

            if discrete:
                # first step: legal-masked argmax (executable); future steps: plain argmax (approx)
                idx = seq.argmax(axis=-1)                       # (E, Hm, A)
                masked0 = seq[:, 0].copy()
                masked0[legal.astype(int) == 0] = -1e9
                idx[:, 0] = masked0.argmax(axis=-1)
                onehot = np.zeros((*idx.shape, num_actions), dtype=np.float32)
                np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
                wm_acts.append(onehot)                         # (E, Hm, A, num_actions)
                first_acts.append(idx[:, 0])                   # (E, A) indices
            else:
                raw = norm.unnormalize(seq, "actions")         # (E, Hm, A, ad) raw
                wm_acts.append(raw)
                first_acts.append(raw[:, 0])

        wm_acts = np.stack(wm_acts, axis=0)                    # (M, E, Hm, A, ad)
        first_acts = np.stack(first_acts, axis=0)              # (M, E, A[, ad])
        if cuda:
            torch.cuda.synchronize()
        t_gen += time.perf_counter() - _tg
        nstep += 1

        if M == 1:
            chosen = first_acts[0]
        elif select == "random":
            best = sel_rng.randint(0, M, size=num_episodes)    # random candidate (control)
            chosen = np.stack([first_acts[best[i], i] for i in range(num_episodes)], axis=0)
        else:
            if cuda:
                torch.cuda.synchronize()
            _ts = time.perf_counter()
            scores = to_np(wm_score(wm, raw_obs, wm_acts, device))   # (M, E)
            if cuda:
                torch.cuda.synchronize()
            t_score += time.perf_counter() - _ts
            best = scores.argmax(axis=0)                       # (E,)
            chosen = np.stack([first_acts[best[i], i] for i in range(num_episodes)], axis=0)

        new = []
        for i in range(num_episodes):
            if dones[i] == 1:
                new.append(obs[i][None])
                continue
            o, r, d, info = envs[i].step(chosen[i])
            new.append(o[None])
            ep_rew[i] = ep_rew[i] + np.asarray(r, dtype=float).reshape(-1)   # broadcast scalar or per-agent
            if i == 0 and t < 3:
                print(f"DBG t={t} mpl={C.max_path_length} r={np.asarray(r).reshape(-1)} d={np.asarray(d).reshape(-1)} act={np.asarray(chosen[i]).reshape(-1)}")
            if getattr(C, "use_return_to_go", False):
                returns[i] = worker._update_return_to_go(returns[i], r)
            if np.asarray(d).all() or t >= C.max_path_length - 1:
                dones[i] = 1
                print(f"DBG ep{i} len={t+1} ep_rew={ep_rew[i]}")
                if isinstance(info, dict) and "battle_won" in info:
                    wins[i] = float(info["battle_won"])
        obs = np.concatenate(new, axis=0)
        if obs_noise_std > 0:
            obs = obs + np.random.randn(*obs.shape).astype(obs.dtype) * obs_noise_std
        t += 1
        env_ts = env_ts + 1

    if nstep > 0:
        per_env = num_episodes  # candidates generated in parallel across envs each step
        print(f"[TIMING] M={M} sel={select} steps={nstep} "
              f"cand_gen={t_gen/nstep*1000:.1f}ms/step "
              f"wm_score={t_score/nstep*1000:.2f}ms/step "
              f"(per-decision per-env: gen={t_gen/nstep/per_env*1000:.2f}ms, score={t_score/nstep/per_env*1000:.3f}ms)")
    return np.array([x.mean() for x in ep_rew]), np.asarray(wins, dtype=float), ep_rew


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-g", "--gpu", default="0")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--n_agents", type=int, required=True)
    p.add_argument("--obs_dim", type=int, required=True)
    p.add_argument("--num_actions", type=int, default=0)
    p.add_argument("--act_dim", type=int, default=0)
    p.add_argument("--discrete", action="store_true")
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--num_eval", type=int, default=16)
    p.add_argument("--steps", type=int, default=5, help="policy denoising steps")
    p.add_argument("--load_step", type=int, default=0, help="checkpoint step (0=auto best)")
    p.add_argument("--ablate_random", action="store_true", help="also run random-select control arm")
    p.add_argument("--test_ret", type=float, default=-1.0, help="override return-conditioning test_ret (>=0 to set)")
    p.add_argument("--n_dyn", type=int, default=8)
    p.add_argument("--n_rew", type=int, default=4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_slots", type=int, default=4)
    p.add_argument("--obs_noise", type=float, default=0.0, help="std of Gaussian obs noise injected at test time")
    p.add_argument("--save_ep", type=str, default="", help="save per-episode rewards to JSON")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    wm_ckpt = args.wm if os.path.isabs(args.wm) else os.path.join(PROJECT_ROOT, args.wm)

    worker = load_worker(run_dir)
    device = worker.Config.device
    load_step = args.load_step if args.load_step > 0 else find_best_step(run_dir)
    load_policy_ckpt(worker, load_step)
    if args.test_ret >= 0:
        print(f"Override test_ret: {worker.Config.test_ret} -> {args.test_ret}")
        worker.Config.test_ret = args.test_ret
    worker.Config.num_eval = args.num_eval
    worker.Config.num_envs = min(args.num_eval, worker.Config.num_envs)
    # match the policy denoising budget used elsewhere
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    wm_act_dim = args.num_actions if args.discrete else args.act_dim
    wm = load_wm(wm_ckpt, args.n_agents, args.obs_dim, wm_act_dim, device,
                 args.n_dyn, args.n_rew, args.hidden, args.n_slots)

    print(f"=== run_dir={run_dir}")
    print(f"=== WM={wm_ckpt}  steps={args.steps}  H={args.H}  num_eval={args.num_eval}")
    num_envs = worker.Config.num_envs
    base_seed = getattr(worker.Config, "seed", 0)
    if args.M == 1:
        arms = [("policy-only", 1, "wm")]
    elif args.ablate_random:
        arms = [("policy-only", 1, "wm"),
                (f"rand-sel(M={args.M})", args.M, "random"),
                (f"WM-sel(M={args.M})", args.M, "wm")]
    else:
        arms = [("policy-only", 1, "wm"), (f"WM-plan(M={args.M})", args.M, "wm")]
    for tag, M, sel in arms:
        set_seed(base_seed)   # paired: identical env-reset scenarios across arms
        rews, wins, all_ep_rews = [], [], []
        remaining = args.num_eval
        while remaining > 0:
            ne = min(remaining, num_envs)
            r_arr, w_arr, ep_rews_batch = planning_eval(worker, wm, ne, M, args.H, device,
                                         num_actions=args.num_actions, discrete=args.discrete, select=sel,
                                         obs_noise_std=args.obs_noise)
            all_ep_rews.extend([x.mean() for x in ep_rews_batch])
            rews.extend(r_arr.tolist())
            wins.extend(w_arr.tolist())
            remaining -= ne
        print(f"[RESULT] {tag}  mean_ep_reward={np.mean(rews):.3f}  win_rate={np.mean(wins):.3f}  std={np.std(all_ep_rews):.3f}  p10={np.percentile(all_ep_rews,10):.3f}  (n={len(rews)})")
        if args.save_ep:
            import json
            ep_out = {"tag": tag, "rews": all_ep_rews, "wins": wins, "obs_noise": args.obs_noise}
            out_path = args.save_ep if args.save_ep.endswith('.json') else f"{args.save_ep}_{tag.replace('/','-')}.json"
            json.dump(ep_out, open(out_path, 'w'))
            print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
