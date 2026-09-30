import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from run_scripts.plan_eval import (
    find_best_step,
    load_policy_ckpt,
    load_worker,
    planning_eval,
    set_seed,
)
from world_model import MoEWorldModel
from world_model.baseline_world_model import BaselineWorldModel
from world_model.routing_variant_world_model import RoutingVariantWorldModel


def load_scorer(args, device):
    ckpt_path = args.wm if os.path.isabs(args.wm) else os.path.join(PROJECT_ROOT, args.wm)
    act_dim = args.num_actions if args.discrete else args.act_dim
    if args.scorer in {"moe", "soft_sparse"}:
        model = MoEWorldModel(
            n_agents=args.n_agents,
            obs_dim=args.obs_dim,
            act_dim=act_dim,
            n_dyn_experts=args.n_dyn,
            n_rew_experts=args.n_rew,
            hidden_dim=args.hidden,
            n_slots=args.n_slots,
        )
    elif args.scorer in {"soft_only", "sparse_only", "sparse_soft"}:
        model = RoutingVariantWorldModel(
            n_agents=args.n_agents,
            obs_dim=args.obs_dim,
            act_dim=act_dim,
            n_dyn_experts=args.n_dyn,
            n_rew_experts=args.n_rew,
            hidden_dim=args.hidden,
            n_slots=args.n_slots,
            variant=args.scorer,
        )
    else:
        model = BaselineWorldModel(
            n_agents=args.n_agents,
            obs_dim=args.obs_dim,
            act_dim=act_dim,
            hidden_dim=args.hidden,
            model_type=args.scorer,
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wm", required=True)
    parser.add_argument(
        "--scorer",
        required=True,
        choices=["moe", "soft_sparse", "soft_only", "sparse_only", "sparse_soft", "monolithic", "independent"],
    )
    parser.add_argument("--n_agents", type=int, required=True)
    parser.add_argument("--obs_dim", type=int, required=True)
    parser.add_argument("--num_actions", type=int, default=0)
    parser.add_argument("--act_dim", type=int, default=0)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--test_ret", type=float, default=-1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_dyn", type=int, default=8)
    parser.add_argument("--n_rew", type=int, default=4)
    parser.add_argument("--n_slots", type=int, default=4)
    parser.add_argument("--select", choices=["wm", "random"], default="wm")
    parser.add_argument("--save_ep", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    worker = load_worker(run_dir)
    device = worker.Config.device
    load_step = args.load_step if args.load_step > 0 else find_best_step(run_dir)
    load_policy_ckpt(worker, load_step)
    if args.test_ret >= 0:
        print(f"Override test_ret: {worker.Config.test_ret} -> {args.test_ret}")
        worker.Config.test_ret = args.test_ret
    worker.Config.num_eval = args.num_eval
    worker.Config.num_envs = min(args.num_eval, worker.Config.num_envs)
    worker.trainer.ema_model.max_denoising_steps = int(args.steps)
    if hasattr(worker.trainer, "model"):
        worker.trainer.model.max_denoising_steps = int(args.steps)

    scorer, ckpt_path = load_scorer(args, device)
    n_params = sum(p.numel() for p in scorer.parameters())
    print(f"=== run_dir={run_dir}")
    print(f"=== scorer={args.scorer} params={n_params:,}")
    print(f"=== WM={ckpt_path} steps={args.steps} H={args.H} M={args.M} num_eval={args.num_eval}")

    base_seed = getattr(worker.Config, "seed", 0)
    num_envs = worker.Config.num_envs
    set_seed(base_seed)
    all_rews, all_wins, all_ep_rews = [], [], []
    remaining = args.num_eval
    while remaining > 0:
        ne = min(remaining, num_envs)
        rews, wins, ep_rews = planning_eval(
            worker,
            scorer,
            ne,
            args.M,
            args.H,
            device,
            num_actions=args.num_actions,
            discrete=args.discrete,
            select=args.select,
        )
        all_rews.extend(rews.tolist())
        all_wins.extend(wins.tolist())
        all_ep_rews.extend([x.mean() for x in ep_rews])
        remaining -= ne

    tag = f"{args.scorer}-sel(M={args.M})" if args.select == "wm" else f"rand-sel(M={args.M})"
    print(
        f"[RESULT] {tag}  mean_ep_reward={np.mean(all_rews):.3f}  "
        f"win_rate={np.mean(all_wins):.3f}  std={np.std(all_ep_rews):.3f}  "
        f"p10={np.percentile(all_ep_rews, 10):.3f}  (n={len(all_rews)})"
    )
    if args.save_ep:
        out = {
            "tag": tag,
            "rews": all_ep_rews,
            "wins": all_wins,
            "scorer": args.scorer,
            "wm": ckpt_path,
            "M": args.M,
            "H": args.H,
            "load_step": load_step,
        }
        out_path = args.save_ep if args.save_ep.endswith(".json") else f"{args.save_ep}_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
