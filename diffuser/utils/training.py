import copy
import os
from typing import Optional, Dict, Any

import numpy as np

import einops
import torch
from ml_logger import logger

import diffuser

from .arrays import apply_dict, batch_to_device, to_device, to_np
from .timer import Timer
from .gradient_monitor import GradientMonitor


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        gradient_clip_norm=None,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=50000,
        label_freq=100000,
        eval_freq=50000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device="cuda",
        save_checkpoints=False,
        enable_gradient_monitor_console=False,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        wandb_entity=None,
        wandb_tags=None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema

        if save_freq and eval_freq and save_freq != 0 and eval_freq != 0:
            if (eval_freq % save_freq != 0) and (save_freq % eval_freq != 0):
                logger.print(
                    f"[ utils/training ] Warning: eval_freq ({eval_freq}) and save_freq ({save_freq}) are not multiples of each other."
                )
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.eval_freq = eval_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.gradient_clip_norm = gradient_clip_norm

        self.dataset = dataset
        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
            self.dataloader_vis = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )

        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.evaluator = None
        self.device = train_device

        # 初始化梯度监控器
        self.gradient_monitor = GradientMonitor(
            log_freq=log_freq,
            gradient_clip_threshold=10.0,
            gradient_vanish_threshold=1e-7,
            track_parameter_changes=True,
            console_logging=enable_gradient_monitor_console,
        )

        self.use_wandb = use_wandb
        self._wandb = None
        self._wandb_run = None
        if self.use_wandb:
            default_config = {
                "train_batch_size": train_batch_size,
                "train_lr": train_lr,
                "ema_decay": ema_decay,
                "gradient_accumulate_every": gradient_accumulate_every,
                "gradient_clip_norm": gradient_clip_norm,
                "log_freq": log_freq,
                "save_freq": save_freq,
                "eval_freq": eval_freq,
            }
            if wandb_config:
                default_config.update(wandb_config)
            self._init_wandb(
                project=wandb_project,
                run_name=wandb_run_name,
                entity=wandb_entity,
                tags=wandb_tags,
                config=default_config,
            )

    def _init_wandb(self, project=None, run_name=None, entity=None, tags=None, config=None):
        """Initialize Weights & Biases logging"""
        import os
        import multiprocessing

        # Skip wandb initialization in worker processes
        if multiprocessing.current_process().name != 'MainProcess':
            print(f"Skipping wandb init in process: {multiprocessing.current_process().name}")
            return

        try:
            import wandb
            print(f"Initializing wandb with project='{project}', run_name='{run_name}'")
            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                entity=entity,
                tags=tags,
                config=config,
                reinit=True,
            )
            if self._wandb_run is None:
                raise RuntimeError("wandb.init returned None")
            print(f"✅ Wandb initialized successfully! Run URL: {self._wandb_run.url}")
        except Exception as e:
            print(f"❌ Error: Failed to initialize wandb: {e}")
            print("Terminating training since wandb logging is required.")
            raise SystemExit(1)

    def _wandb_log(self, metrics, step=None):
        """Log metrics to wandb if available"""
        try:
            import wandb
            if wandb.run is not None:
                # Convert tensor values to float for logging
                log_metrics = {}
                for key, value in metrics.items():
                    if hasattr(value, 'item'):
                        log_metrics[key] = value.item()
                    else:
                        log_metrics[key] = value
                wandb.log(log_metrics, step=step)
        except Exception as e:
            pass  # Silently skip logging errors

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def _prepare_eval_metrics_for_wandb(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested evaluation metrics into a wandb-friendly dict."""

        def _collect(key_prefix: str, value: Any, output: Dict[str, float]):
            if value is None:
                return
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _collect(f"{key_prefix}/{sub_key}", sub_value, output)
                return
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return
                if value.ndim == 0:
                    output[key_prefix] = float(value)
                else:
                    for idx, item in np.ndenumerate(value):
                        idx_key = "/".join(map(str, idx))
                        output[f"{key_prefix}/{idx_key}"] = float(item)
                return
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return
                for idx, item in enumerate(value):
                    _collect(f"{key_prefix}/{idx}", item, output)
                return
            if hasattr(value, "item"):
                try:
                    output[key_prefix] = float(value.item())
                except (TypeError, ValueError):
                    pass
                return
            try:
                output[key_prefix] = float(value)
            except (TypeError, ValueError):
                pass

        flattened: Dict[str, float] = {}
        for metric_key, metric_value in metrics.items():
            _collect(f"eval/{metric_key}", metric_value, flattened)
        return flattened

    def finish_training(self):
        if self.step % self.save_freq == 0:
            self.save()
        if self.eval_freq > 0 and self.step % self.eval_freq == 0:
            self.evaluate()

        # 输出梯度监控总结
        logger.print("\n" + "="*60)
        logger.print("GRADIENT MONITORING SUMMARY")
        logger.print("="*60)

        summary_stats = self.gradient_monitor.get_summary_stats()
        for key, value in summary_stats.items():
            logger.print(f"{key}: {value:.6f}")

        if self.use_wandb and summary_stats:
            wandb_summary = {f"summary/{k}": v for k, v in summary_stats.items()}
            self._wandb_log(wandb_summary)

        # 诊断训练问题
        logger.print("\nTRAINING DIAGNOSIS:")
        diagnosis = self.gradient_monitor.diagnose_training_issues()
        for i, issue in enumerate(diagnosis, 1):
            logger.print(f"{i}. {issue}")

        logger.print("="*60)

        if self.evaluator is not None:
            del self.evaluator

        if self.use_wandb and self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()
        for step_idx in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(**batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            # 在优化器更新前监控梯度
            gradient_stats = self.gradient_monitor.track_gradients(self.model, self.step)

            # 梯度裁剪（如果启用）
            if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.eval_freq > 0 and self.step > 0 and self.step % self.eval_freq == 0:
                self.evaluate()

            if self.step % self.log_freq == 0:
                # Create progress bar display
                progress = (step_idx + 1) / n_train_steps
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                infos_str = " | ".join(
                    [f"{key}: {val:.4f}" for key, val in infos.items()]
                )

                # Calculate elapsed and remaining time
                elapsed_time = timer(reset=False)
                if progress > 0:
                    estimated_total_time = elapsed_time / progress
                    remaining_time = estimated_total_time - elapsed_time

                    # Format remaining time
                    if remaining_time > 3600:  # more than 1 hour
                        eta_str = f" | ETA: {remaining_time/3600:.1f}h"
                    elif remaining_time > 60:  # more than 1 minute
                        eta_str = f" | ETA: {remaining_time/60:.1f}m"
                    else:
                        eta_str = f" | ETA: {remaining_time:.0f}s"
                else:
                    eta_str = ""

                # Clean progress bar output focused on key metrics
                logger.print(
                    f"Step {self.step:6d} [{bar}] {progress:6.1%} | Loss: {loss:.4f} | {infos_str} | Time: {timer():.1f}s{eta_str}"
                )
                # Comment out detailed logging to reduce verbose output
                # metrics = {k: v.detach().item() for k, v in infos.items()}
                # logger.log(
                #     step=self.step, loss=loss.detach().item(), **metrics, flush=True
                # )

                if self.use_wandb:
                    wandb_metrics: Dict[str, Any] = {
                        "train/total_loss": loss,
                    }
                    # Log all important training metrics (including loss components)
                    for key, value in infos.items():
                        if 'loss' in key.lower() or 'reward' in key.lower() or 'return' in key.lower():
                            wandb_metrics[f"train/{key}"] = value
                        else:
                            wandb_metrics[f"train/{key}"] = value
                    self._wandb_log(wandb_metrics, step=self.step)

            if self.sample_freq and self.step == 0:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.model.__class__ == diffuser.models.diffusion.GaussianDiffusion:
                    self.inv_render_samples()
                elif self.model.__class__ == diffuser.models.diffusion.ValueDiffusion:
                    pass
                else:
                    self.render_samples()

            self.step += 1

    def evaluate(self):
        assert (
            self.evaluator is not None
        ), "Method `evaluate` can not be called when `self.evaluator` is None. Set evaluator with `self.set_evaluator` first."
        metrics = self.evaluator.evaluate(load_step=self.step)
        if self.use_wandb and isinstance(metrics, dict):
            wandb_metrics = self._prepare_eval_metrics_for_wandb(metrics)
            if wandb_metrics:
                self._wandb_log(wandb_metrics, step=self.step)
        return metrics

    def save(self):
        """
        saves model and ema to disk;
        syncs to storage bucket if a bucket is specified
        """

        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def load(self):
        """
        loads model and ema from disk
        """

        loadpath = os.path.join(self.bucket, logger.prefix, "checkpoint/state.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        """
        renders training points
        """

        # get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        # get trajectories and condition at t=0 from batch
        trajectories = to_np(batch["x"])
        # conditions = to_np(batch.cond[0])[:, None]

        # [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[..., self.dataset.action_dim :]
        shape = normed_observations.shape
        observations = self.dataset.normalizer.unnormalize(
            normed_observations.reshape(-1, *normed_observations.shape[2:]),
            "observations",
        ).reshape(shape)

        savepath = os.path.join("images", "sample-reference.png")
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):
            # get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch["cond"], self.device)
            player_conditions = None
            if (
                "player_idxs" in conditions and "player_hoop_sides" in conditions
            ):  # must have add player info
                player_conditions = {
                    "player_idxs": conditions["player_idxs"],
                    "player_hoop_sides": conditions["player_hoop_sides"],
                }
                conditions = {
                    key: val
                    for key, val in list(conditions.items())
                    if (key != "player_idxs" and key != "player_hoop_sides")
                }

            # repeat each item in conditions `n_samples` times
            if len(list(conditions.values())[0].shape) == 4:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b t a d -> (repeat b) t a d",
                    repeat=n_samples,
                )
            elif len(list(conditions.values())[0].shape) == 3:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b a d -> (repeat b) a d",
                    repeat=n_samples,
                )
            else:
                conditions = apply_dict(
                    einops.repeat,
                    conditions,
                    "b d -> (repeat b) d",
                    repeat=n_samples,
                )

            if player_conditions is not None:
                player_conditions = apply_dict(
                    einops.repeat,
                    player_conditions,
                    "b t a d -> (repeat b) t a d",
                    repeat=n_samples,
                )
                for key, val in list(player_conditions.items()):
                    assert key == "player_idxs" or key == "player_hoop_sides"
                    conditions[key] = val

            # [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(
                    torch.ones(n_samples, 1, self.model.n_agents), self.device
                )
            else:
                returns = None

            samples = self.ema_model.conditional_sample(conditions, returns=returns)
            samples = to_np(samples)

            # Check if model uses inverse dynamics (which returns observations only)
            # or full trajectory (which returns actions + observations)
            if hasattr(self.ema_model, 'use_inv_dyn') and self.ema_model.use_inv_dyn:
                # For inverse dynamics: samples are already observations only
                # [ n_samples x horizon x agent x observation_dim ]
                normed_observations = samples
            else:
                # For full trajectory: need to extract observations from samples
                # [ n_samples x horizon x agent x observation_dim ]
                normed_observations = samples[:, :, :, self.dataset.action_dim :]

            # [ 1 x 1 x agent x observation_dim ]
            # normed_conditions = to_np(batch.cond[0])[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            # [ n_samples x (horizon + 1) x agent x observation_dim ]
            # normed_observations = np.concatenate(
            #     [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
            #     axis=1,
            # )

            # [ n_samples x (horizon + 1) x agent x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            # @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join("images", f"sample-{i}.png")
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        """
        renders samples from (ema) diffusion model
        """
        for i in range(batch_size):
            # get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch["cond"], self.device)
            # repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                "b ... -> (repeat b) ...",
                repeat=n_samples,
            )
            # [ n_samples x horizon x n_agents x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(
                    torch.ones(n_samples, 1, self.model.n_agents), self.device
                )
            else:
                returns = None

            samples = self.ema_model.conditional_sample(conditions, returns=returns)
            samples = to_np(samples)

            # [ n_samples x horizon x n_agents x observation_dim ]
            normed_observations = samples[:, :, :, :]

            # [ 1 x 1 x n_agents x observation_dim ]
            # normed_conditions = to_np(batch.cond[0])[:, None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            # [ n_samples x (horizon + 1) x n_agents x observation_dim ]
            # normed_observations = np.concatenate(
            #     [np.repeat(normed_conditions, n_samples, axis=0), normed_observations],
            #     axis=1,
            # )

            # [ n_samples x (horizon + 1) x n_agents x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(
                normed_observations, "observations"
            )

            savepath = os.path.join("images", f"sample-{i}.png")
            self.renderer.composite(savepath, observations)
