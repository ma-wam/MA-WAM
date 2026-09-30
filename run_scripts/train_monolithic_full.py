"""
R2: Train a FULL Monolithic world model (dynamics + reward head) for fair comparison vs MoE.
Drop-in: BaselineWorldModel.loss has the same signature as MoEWorldModel.loss, so WorldModelTrainer
works unchanged. Saves to logs/phase1_wm_mono/<env>/<split> (does NOT touch the MoE checkpoints).

Usage: python run_scripts/train_monolithic_full.py -c <phase1_cfg>.yaml -g 0 [--model monolithic]
"""
from training_seeds import add_training_arguments, prepare_training, seed_output
import argparse, os, sys
import torch, yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from world_model import WorldModelTrainer
from world_model.baseline_world_model import BaselineWorldModel
from run_scripts.paths import data_path_from_config


def main():
    parser = argparse.ArgumentParser(description="Train full Monolithic/Independent World Model")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-g", "--gpu", type=str, default="0")
    parser.add_argument("--model", choices=["monolithic", "independent"], default="monolithic")
    add_training_arguments(parser)
    args = parser.parse_args()
    if prepare_training(args, None):
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        config = yaml.safe_load(f)
        config["training_seed"] = args.training_seed
    print(f"Phase1 {args.model} WM | config={args.config} | device={device}")

    data_path = data_path_from_config(config, PROJECT_ROOT)
    print("Data path:", data_path)

    discrete_action = config.get("discrete_action", False)
    num_actions = config.get("num_actions", None)
    act_dim = num_actions if discrete_action and num_actions else config["act_dim"]

    # param-matched to MoE (~0.9-1.1M) via hidden_dim (default 512 for monolithic)
    hidden = config.get("mono_hidden_dim", 512)
    world_model = BaselineWorldModel(
        n_agents=config["n_agents"], obs_dim=config["obs_dim"], act_dim=act_dim,
        hidden_dim=hidden, model_type=args.model,
    )
    print(f"{args.model} params: {sum(p.numel() for p in world_model.parameters()):,} (hidden={hidden})")

    trainer = WorldModelTrainer(
        world_model=world_model, data_path=data_path,
        batch_size=config.get("batch_size", 256), lr=config.get("learning_rate", 3e-4),
        device=device, val_ratio=config.get("val_ratio", 0.1),
        use_wandb=config.get("use_wandb", False), wandb_project=config.get("wandb_project", "CoFlow-WM"),
        wandb_run_name=config.get("wandb_run_name", None), wandb_config=config,
        discrete_action=discrete_action, num_actions=num_actions,
    )
    save_dir = os.path.join(PROJECT_ROOT, "logs", f"phase1_wm_{args.model}",
                            config["env_name"], config["data_split"])
    save_dir = seed_output(save_dir, args)
    trainer.train(n_steps=config.get("n_train_steps", 40000), save_dir=save_dir,
                  save_freq=config.get("save_freq", 5000), log_freq=config.get("log_freq", 500))
    print(f"Done. Saved to: {save_dir}")


if __name__ == "__main__":
    main()
