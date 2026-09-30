from training_seeds import add_training_arguments, prepare_training, seed_output
import argparse
import os
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from world_model.routing_variant_world_model import RoutingVariantWorldModel
from world_model import WorldModelTrainer
from run_scripts.paths import data_path_from_config


def resolve_data_path(config):
    return data_path_from_config(config, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Train routing-variant world model")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-g", "--gpu", default="0")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["soft_sparse", "soft_only", "sparse_only", "sparse_soft"],
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--save_root", default="logs/routing_variant_wm")
    add_training_arguments(parser)
    args = parser.parse_args()
    if prepare_training(args, None):
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config["training_seed"] = args.training_seed

    discrete_action = config.get("discrete_action", False)
    num_actions = config.get("num_actions", None)
    act_dim = num_actions if discrete_action and num_actions else config["act_dim"]
    data_path = resolve_data_path(config)

    print("=" * 70)
    print(f"Training routing-variant WM: {args.variant}")
    print(f"Config: {args.config}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")

    world_model = RoutingVariantWorldModel(
        n_agents=config["n_agents"],
        obs_dim=config["obs_dim"],
        act_dim=act_dim,
        n_dyn_experts=config.get("n_dyn_experts", 8),
        n_rew_experts=config.get("n_rew_experts", 4),
        hidden_dim=config.get("hidden_dim", 256),
        n_slots=config.get("n_slots", 4),
        variant=args.variant,
    )
    n_params = sum(p.numel() for p in world_model.parameters())
    print(f"World model parameters: {n_params:,}")

    trainer = WorldModelTrainer(
        world_model=world_model,
        data_path=data_path,
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 3e-4),
        device=device,
        val_ratio=config.get("val_ratio", 0.1),
        use_wandb=config.get("use_wandb", False),
        wandb_project=config.get("wandb_project", "CoFlow-WM"),
        wandb_run_name=f"{config.get('wandb_run_name', 'wm')}-{args.variant}",
        wandb_config={**config, "routing_variant": args.variant},
        discrete_action=discrete_action,
        num_actions=num_actions,
    )

    save_dir = os.path.join(
        PROJECT_ROOT,
        args.save_root,
        args.variant,
        config["env_name"],
        config["data_split"],
    )
    save_dir = seed_output(save_dir, args)
    trainer.train(
        n_steps=args.steps or config.get("n_train_steps", 40000),
        save_dir=save_dir,
        save_freq=config.get("save_freq", 5000),
        log_freq=config.get("log_freq", 500),
    )
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
