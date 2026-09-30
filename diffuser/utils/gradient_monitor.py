"""
梯度监控工具，用于检测梯度消失、爆炸等问题
"""

import torch
import numpy as np
import logging
from ml_logger import logger
from typing import Dict, List, Optional

class GradientMonitor:
    """监控模型训练过程中的梯度信息"""

    def __init__(
        self,
        log_freq: int = 100,
        gradient_clip_threshold: float = 10.0,
        gradient_vanish_threshold: float = 1e-7,
        track_parameter_changes: bool = True,
        console_logging: bool = True,
    ):
        """
        初始化梯度监控器

        Args:
            log_freq: 记录频率
            gradient_clip_threshold: 梯度爆炸阈值
            gradient_vanish_threshold: 梯度消失阈值
            track_parameter_changes: 是否跟踪参数变化
        """
        self.log_freq = log_freq
        self.gradient_clip_threshold = gradient_clip_threshold
        self.gradient_vanish_threshold = gradient_vanish_threshold
        self.track_parameter_changes = track_parameter_changes
        self.console_logging = console_logging

        # 统计信息
        self.gradient_norms = []
        self.parameter_norms = []
        self.parameter_changes = []
        self.exploding_grad_count = 0
        self.vanishing_grad_count = 0
        self.step_count = 0

        # 存储上一步的参数（用于计算变化量）
        self.prev_params = {}

    def track_gradients(self, model: torch.nn.Module, step: int) -> Dict[str, float]:
        """
        跟踪模型梯度信息

        Args:
            model: 要监控的模型
            step: 当前训练步数

        Returns:
            包含梯度统计信息的字典
        """
        self.step_count = step

        # 计算梯度范数
        total_norm = 0.0
        param_count = 0
        gradient_dict = {}
        layer_grad_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

                # 记录每层的梯度范数
                layer_name = name.split('.')[0] if '.' in name else name
                if layer_name not in layer_grad_norms:
                    layer_grad_norms[layer_name] = []
                layer_grad_norms[layer_name].append(param_norm)

        total_norm = total_norm ** (1. / 2)

        # 检测梯度问题
        if total_norm > self.gradient_clip_threshold:
            self.exploding_grad_count += 1
            gradient_dict['gradient_explosion'] = 1.0
        else:
            gradient_dict['gradient_explosion'] = 0.0

        if total_norm < self.gradient_vanish_threshold:
            self.vanishing_grad_count += 1
            gradient_dict['gradient_vanishing'] = 1.0
        else:
            gradient_dict['gradient_vanishing'] = 0.0

        # 存储统计信息
        self.gradient_norms.append(total_norm)
        gradient_dict.update({
            'gradient_norm': total_norm,
            'param_count_with_grad': param_count,
            'exploding_grad_ratio': self.exploding_grad_count / (step + 1),
            'vanishing_grad_ratio': self.vanishing_grad_count / (step + 1)
        })

        # 记录每层梯度范数统计
        for layer_name, norms in layer_grad_norms.items():
            avg_norm = np.mean(norms)
            max_norm = np.max(norms)
            gradient_dict[f'{layer_name}_avg_grad_norm'] = avg_norm
            gradient_dict[f'{layer_name}_max_grad_norm'] = max_norm

        # 跟踪参数变化
        if self.track_parameter_changes:
            param_stats = self.track_parameter_changes_fn(model)
            gradient_dict.update(param_stats)

        # 定期记录详细信息
        if step % self.log_freq == 0:
            self.log_gradient_stats(gradient_dict, step)

        return gradient_dict

    def track_parameter_changes_fn(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        跟踪参数变化量

        Args:
            model: 要监控的模型

        Returns:
            参数变化统计信息
        """
        param_stats = {}
        total_param_norm = 0.0
        total_change_norm = 0.0
        param_count = 0

        current_params = {}

        for name, param in model.named_parameters():
            if param.data is not None:
                current_params[name] = param.data.clone()
                param_norm = param.data.norm(2).item()
                total_param_norm += param_norm ** 2
                param_count += 1

                # 计算参数变化量
                if name in self.prev_params:
                    change = param.data - self.prev_params[name]
                    change_norm = change.norm(2).item()
                    total_change_norm += change_norm ** 2

        total_param_norm = total_param_norm ** (1. / 2)
        total_change_norm = total_change_norm ** (1. / 2)

        # 存储统计信息
        self.parameter_norms.append(total_param_norm)
        self.parameter_changes.append(total_change_norm)

        param_stats.update({
            'parameter_norm': total_param_norm,
            'parameter_change_norm': total_change_norm,
            'param_change_ratio': total_change_norm / (total_param_norm + 1e-8)
        })

        # 更新上一步参数
        self.prev_params = current_params

        return param_stats

    def log_gradient_stats(self, gradient_dict: Dict[str, float], step: int):
        """记录梯度统计信息"""

        # 打印到控制台
        if self.console_logging:
            logger.print(f"Step {step} Gradient Stats:")
            logger.print(f"  Gradient Norm: {gradient_dict['gradient_norm']:.6f}")
            logger.print(f"  Gradient Explosion: {gradient_dict['gradient_explosion']}")
            logger.print(f"  Gradient Vanishing: {gradient_dict['gradient_vanishing']}")

            if self.track_parameter_changes and 'parameter_norm' in gradient_dict:
                logger.print(f"  Parameter Norm: {gradient_dict['parameter_norm']:.6f}")
                logger.print(f"  Parameter Change: {gradient_dict['parameter_change_norm']:.6f}")
                logger.print(f"  Change Ratio: {gradient_dict['param_change_ratio']:.6f}")

        # 记录到日志文件（允许关闭所有输出）
        if self.console_logging:
            logger.log(step=step, **gradient_dict, flush=True)

    def get_summary_stats(self) -> Dict[str, float]:
        """获取梯度监控的总结统计信息"""
        if not self.gradient_norms:
            return {}

        summary = {
            'avg_gradient_norm': np.mean(self.gradient_norms),
            'max_gradient_norm': np.max(self.gradient_norms),
            'min_gradient_norm': np.min(self.gradient_norms),
            'gradient_norm_std': np.std(self.gradient_norms),
            'total_exploding_grads': self.exploding_grad_count,
            'total_vanishing_grads': self.vanishing_grad_count,
            'exploding_grad_percentage': (self.exploding_grad_count / len(self.gradient_norms)) * 100,
            'vanishing_grad_percentage': (self.vanishing_grad_count / len(self.gradient_norms)) * 100
        }

        if self.parameter_norms:
            summary.update({
                'avg_parameter_norm': np.mean(self.parameter_norms),
                'max_parameter_change': np.max(self.parameter_changes) if self.parameter_changes else 0,
                'avg_parameter_change': np.mean(self.parameter_changes) if self.parameter_changes else 0
            })

        return summary

    def reset_stats(self):
        """重置所有统计信息"""
        self.gradient_norms = []
        self.parameter_norms = []
        self.parameter_changes = []
        self.exploding_grad_count = 0
        self.vanishing_grad_count = 0
        self.step_count = 0
        self.prev_params = {}

    def diagnose_training_issues(self) -> List[str]:
        """诊断训练中可能存在的问题"""
        issues = []

        if not self.gradient_norms:
            return ["No gradient data available for diagnosis"]

        # 检查梯度爆炸
        exploding_ratio = (self.exploding_grad_count / len(self.gradient_norms)) * 100
        if exploding_ratio > 10:  # 超过10%的步数出现梯度爆炸
            issues.append(f"Gradient explosion detected in {exploding_ratio:.1f}% of steps. Consider gradient clipping or reducing learning rate.")

        # 检查梯度消失
        vanishing_ratio = (self.vanishing_grad_count / len(self.gradient_norms)) * 100
        if vanishing_ratio > 20:  # 超过20%的步数出现梯度消失
            issues.append(f"Gradient vanishing detected in {vanishing_ratio:.1f}% of steps. Consider using residual connections, different activation functions, or increasing learning rate.")

        # 检查梯度变化趋势
        if len(self.gradient_norms) > 100:
            recent_norms = self.gradient_norms[-100:]
            early_norms = self.gradient_norms[:100]

            recent_avg = np.mean(recent_norms)
            early_avg = np.mean(early_norms)

            if recent_avg < early_avg * 0.1:  # 梯度范数显著减小
                issues.append("Gradient norms are decreasing significantly over time, indicating potential training instability.")

        # 检查参数变化
        if self.parameter_changes and len(self.parameter_changes) > 10:
            recent_changes = np.mean(self.parameter_changes[-10:])
            if recent_changes < 1e-8:
                issues.append("Parameter changes are very small, model may have stopped learning.")

        if not issues:
            issues.append("No obvious gradient-related issues detected.")

        return issues
