"""
MoE 专家路由可视化工具

生成论文用的图表：
- 路由热力图 (Agent-Expert 亲和度)
- Agent 交互网络图
- 状态空间聚类 (按主导 Expert)
- 反事实消融柱状图
- 路由动态随轨迹变化图
"""

import os
import numpy as np

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class MoEVisualizer:
    """MoE 专家路由可视化工具"""

    def __init__(self, save_dir='figures', dpi=150):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)

    def plot_routing_heatmap(self, affinity_matrix, title='Agent-Expert Affinity',
                              agent_names=None, expert_names=None, filename='routing_heatmap.pdf'):
        """
        绘制 Agent-Expert 亲和度热力图。

        Args:
            affinity_matrix: (n_agents, n_experts)
            agent_names: agent 名称列表
            expert_names: expert 名称列表
        """
        n_agents, n_experts = affinity_matrix.shape
        if agent_names is None:
            agent_names = [f'Agent {i}' for i in range(n_agents)]
        if expert_names is None:
            expert_names = [f'E{i}' for i in range(n_experts)]

        # Square cells when agents >= 4; 2:1 (h:w) when fewer agents
        use_equal = n_agents >= 4
        if use_equal:
            cell_aspect = 'equal'
            fig_h = max(4, n_agents * 1.5)
        else:
            cell_aspect = n_experts / (n_agents * 2)  # makes each cell h:w = 2:1
            fig_h = max(4, n_agents * 2.0)
        fig, ax = plt.subplots(figsize=(max(8, n_experts * 1.0), fig_h))
        im = ax.imshow(affinity_matrix, cmap='YlOrRd', aspect=cell_aspect)

        ax.set_xticks(range(n_experts))
        ax.set_xticklabels(expert_names, rotation=0, ha='center', fontsize=14.5)
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(agent_names, fontsize=14.5)
        ax.set_xlabel('Expert', fontsize=16.5)
        ax.set_ylabel('Agent', fontsize=16.5)
        ax.set_title(title, fontsize=18.5, pad=6)

        # 数值标注: 只有最大值用白色，其余用黑色
        max_val = affinity_matrix.max()
        for i in range(n_agents):
            for j in range(n_experts):
                is_max = affinity_matrix[i, j] == max_val
                val = affinity_matrix[i, j]
                label = f'.{val:.3f}'[1:]  # e.g. 0.122 -> .122
                ax.text(j, i, label,
                        ha='center', va='center', fontsize=14,
                        color='white' if is_max else 'black')

        # Match colorbar height to heatmap content
        if use_equal:
            cbar = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02, aspect=23)
        else:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=17)
        fig.tight_layout(pad=0.5)
        fig.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {os.path.join(self.save_dir, filename)}')

    def plot_interaction_graph(self, interaction_matrix, agent_names=None,
                                filename='interaction_graph.pdf'):
        """
        绘制 Agent 间交互网络图。

        节点 = Agent, 边粗细 = 耦合强度 (通过共享 Expert Slot)。
        """
        n_agents = interaction_matrix.shape[0]
        if agent_names is None:
            agent_names = [f'Agent {i}' for i in range(n_agents)]

        fig, ax = plt.subplots(figsize=(5, 5))

        angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
        radius = 1.0
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius

        offdiag = interaction_matrix.copy()
        np.fill_diagonal(offdiag, 0)
        max_w = offdiag.max() if offdiag.max() > 0 else 1.0

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                w = offdiag[i, j]
                if w > max_w * 0.1:
                    linewidth = 2 + 6 * (w / max_w)
                    alpha = 0.3 + 0.7 * (w / max_w)
                    ax.plot(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        'b-', linewidth=linewidth, alpha=alpha,
                    )
                    mid = (positions[i] + positions[j]) / 2
                    ax.text(mid[0], mid[1], f'{w:.3f}', fontsize=14,
                            ha='center', va='center', color='blue', alpha=0.8)

        for i in range(n_agents):
            circle = plt.Circle(positions[i], 0.3, color='steelblue', ec='black', lw=2.5, zorder=5)
            ax.add_patch(circle)
            ax.text(positions[i, 0], positions[i, 1], agent_names[i],
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=6)

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.set_title('Agent Interaction', fontsize=18.5, pad=10)
        ax.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {os.path.join(self.save_dir, filename)}')

    def plot_expert_state_clusters(self, observations, dominant_expert, n_experts,
                                    filename='expert_state_clusters.pdf'):
        """
        使用 PCA 降维后按主导 Expert 着色的散点图。

        Args:
            observations: (n_samples, n_agents, obs_dim)
            dominant_expert: (n_samples, n_agents)
            n_experts: 专家总数
        """
        from sklearn.decomposition import PCA

        # 使用 agent 0 的观测进行可视化
        obs_flat = observations[:, 0]  # (n_samples, obs_dim)
        labels = dominant_expert[:, 0]  # (n_samples,)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(obs_flat)

        fig, ax = plt.subplots(figsize=(7, 7))
        cmap = plt.cm.get_cmap('tab10', n_experts)

        for e in range(n_experts):
            mask = labels == e
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(e)],
                          label=f'E{e} ({mask.sum()})', alpha=0.5, s=15)

        # Square plot but each axis fits its own data
        ax.set_box_aspect(1)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=14.5)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=14.5)
        ax.tick_params(labelsize=16)
        ax.legend(markerscale=3, fontsize=12, ncol=4,
                  loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  frameon=True, edgecolor='gray')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        fig.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {os.path.join(self.save_dir, filename)}')

    def plot_ablation_degradation(self, ablation_results, filename='expert_ablation.pdf', ylim_max=None):
        """
        反事实消融柱状图: 每个 Expert 被移除后预测精度的下降。

        Args:
            ablation_results: dict from counterfactual_expert_ablation()
            ylim_max: 固定纵轴上限，None则自动
        """
        degradation = ablation_results['degradation']
        baseline = ablation_results['baseline_mse']
        n_experts = len(degradation)

        fig, ax = plt.subplots(figsize=(max(8, n_experts * 1.0), 5))

        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in degradation]
        bars = ax.bar(range(n_experts), degradation, color=colors, edgecolor='black', linewidth=0.5)

        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xlabel('Expert Index', fontsize=16.5)
        ax.set_ylabel('MSE Degradation', fontsize=16.5)
        ax.set_title(f'Expert Ablation (baseline MSE = {baseline:.4f})', fontsize=18.5, pad=10)
        ax.set_xticks(range(n_experts))
        ax.set_xticklabels([f'E{i}' for i in range(n_experts)], fontsize=14.5)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(bottom=0)
        if ylim_max is not None:
            ax.set_ylim(top=ylim_max)
        else:
            ax.set_ylim(top=max(degradation) * 1.15)

        # 数值标注
        for bar, d in zip(bars, degradation):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{d:.3f}', ha='center', va='bottom' if d > 0 else 'top', fontsize=18.5)

        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {os.path.join(self.save_dir, filename)}')

    def plot_routing_dynamics_over_episode(self, obs_seq, act_seq, world_model,
                                            filename='routing_dynamics.pdf'):
        """
        可视化单条轨迹中路由权重随时间的变化。

        Args:
            obs_seq: (T, n_agents, obs_dim) - 单条轨迹的观测序列
            act_seq: (T, n_agents, act_dim) - 单条轨迹的动作序列
            world_model: MoEWorldModel
        """
        T = len(obs_seq)
        n_experts = world_model.dynamics.n_experts
        n_slots = world_model.dynamics.n_slots
        n_agents = world_model.dynamics.n_agents

        # 逐步提取路由权重
        expert_weights_over_time = np.zeros((T, n_agents, n_experts))

        world_model.eval()
        with torch.no_grad():
            for t in range(T):
                obs = torch.tensor(obs_seq[t:t+1], dtype=torch.float32, device=next(world_model.parameters()).device)
                act = torch.tensor(act_seq[t:t+1], dtype=torch.float32, device=obs.device)

                _, routing = world_model.dynamics.forward_with_routing(obs, act)
                combine = routing['combine_probs'].cpu().numpy()[0]  # (n_agents, total_slots)

                for e in range(n_experts):
                    s, end = e * n_slots, (e + 1) * n_slots
                    expert_weights_over_time[t, :, e] = combine[:, s:end].sum(axis=1)

        # 绘图: 3行2列网格，最后一行放图例
        nrows = (n_agents + 1) // 2  # agent子图行数
        fig, axes_all = plt.subplots(nrows + 1, 2, figsize=(10, 3.5 * nrows + 1.2),
                                      gridspec_kw={'height_ratios': [1]*nrows + [0.15]})

        cmap = plt.cm.get_cmap('tab10', n_experts)
        lines = []

        for a in range(n_agents):
            r, c = a // 2, a % 2
            ax = axes_all[r][c]
            for e in range(n_experts):
                ln, = ax.plot(range(T), expert_weights_over_time[:, a, e],
                       color=cmap(e), alpha=0.8, linewidth=1.2)
                if a == 0:
                    lines.append(ln)
            ax.set_title(f'Agent {a}', fontsize=14.5)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=18)
            if c == 0:
                ax.set_ylabel('Weight', fontsize=18.5)
            if r == nrows - 1 or (a + 2 >= n_agents):
                ax.set_xlabel('Timestep', fontsize=18.5)

        # 隐藏多余的子图
        for a in range(n_agents, nrows * 2):
            r, c = a // 2, a % 2
            axes_all[r][c].axis('off')

        # 最后一行全部关闭，用来放图例
        for c in range(2):
            axes_all[nrows][c].axis('off')

        labels = [f'E{e}' for e in range(n_experts)]
        fig.legend(lines, labels, loc='lower center', ncol=n_experts,
                   fontsize=14.5, bbox_to_anchor=(0.5, 0.01),
                   frameon=True, edgecolor='gray')
        fig.suptitle('Expert Routing Dynamics', fontsize=18.5, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {os.path.join(self.save_dir, filename)}')
