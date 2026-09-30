"""
Phase 1: 训练 MoE 世界模型

用法:
    python run_scripts/train_world_model.py -c exp_specs/phase1_wm/mamujoco/2halfcheetah/wm_2halfcheetah_good.yaml -g 0
"""

import argparse
import os
import sys

import torch
import yaml

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from world_model import MoEWorldModel, WorldModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train MoE World Model")
    parser.add_argument("-c", "--config", required=True, help="Config YAML path")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU id")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Phase 1: Training MoE World Model")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")

    # 构建数据路径
    if "data_root" in config:
        # 自定义数据路径（如转换后的 MPE 数据）。相对路径以发布仓库根目录为准。
        data_root = os.path.expanduser(config["data_root"])
        if not os.path.isabs(data_root):
            data_root = os.path.join(PROJECT_ROOT, data_root)
        data_path = os.path.join(data_root, config["env_name"], config["data_split"])
    else:
        # 默认: 发布仓库内的 CoFlow 数据目录。
        m2flow_data_root = os.path.join(PROJECT_ROOT, "diffuser", "datasets", "data")
        data_path = os.path.join(
            m2flow_data_root, config["env_type"], config["env_name"], config["data_split"]
        )
    print(f"Data path: {data_path}")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}. "
            "See README.md for the expected release layout."
        )

    # 离散动作处理: act_dim 使用 one-hot 编码后的维度
    discrete_action = config.get("discrete_action", False)
    num_actions = config.get("num_actions", None)
    act_dim = num_actions if discrete_action and num_actions else config["act_dim"]

    # 创建世界模型
    world_model = MoEWorldModel(
        n_agents=config["n_agents"],
        obs_dim=config["obs_dim"],
        act_dim=act_dim,
        n_dyn_experts=config.get("n_dyn_experts", 8),
        n_rew_experts=config.get("n_rew_experts", 4),
        hidden_dim=config.get("hidden_dim", 256),
        n_slots=config.get("n_slots", 4),
    )

    n_params = sum(p.numel() for p in world_model.parameters())
    print(f"World model parameters: {n_params:,}")

    # 训练
    trainer = WorldModelTrainer(
        world_model=world_model,
        data_path=data_path,
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 3e-4),
        device=device,
        val_ratio=config.get("val_ratio", 0.1),
        use_wandb=config.get("use_wandb", False),
        wandb_project=config.get("wandb_project", "CoFlow-WM"),
        wandb_run_name=config.get("wandb_run_name", None),
        wandb_config=config,
        discrete_action=discrete_action,
        num_actions=num_actions,
    )

    save_dir = os.path.join(
        PROJECT_ROOT, "logs", "phase1_wm",
        config["env_name"], config["data_split"]
    )

    trainer.train(
        n_steps=config.get("n_train_steps", 40000),
        save_dir=save_dir,
        save_freq=config.get("save_freq", 5000),
        log_freq=config.get("log_freq", 500),
    )

    print(f"\nPhase 1 complete. Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
