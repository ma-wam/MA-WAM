import argparse
import csv
import glob
import json
import os
import re

import numpy as np


TASK_ORDER = []
for env in ["simple_spread", "simple_tag", "simple_world"]:
    for split in ["expert", "medium-replay", "medium", "random"]:
        TASK_ORDER.append(("MPE", env, split))
for env in ["2ant", "4ant"]:
    for split in ["Good", "Medium", "Poor"]:
        TASK_ORDER.append(("MAMuJoCo", env, split))
for env in ["3m", "2s3z", "5m_vs_6m", "8m"]:
    for split in ["Good", "Medium", "Poor"]:
        TASK_ORDER.append(("SMAC", env, split))

PRETTY_TASK = {
    "simple_spread": "Spread",
    "simple_tag": "Tag",
    "simple_world": "World",
    "medium-replay": "Md-Replay",
}

PAPER_TABLE_RESULTS = {
    ("MPE", "simple_spread", "expert"): (119.9, 124.8),
    ("MPE", "simple_spread", "medium-replay"): (40.2, 73.3),
    ("MPE", "simple_spread", "medium"): (50.3, 83.0),
    ("MPE", "simple_spread", "random"): (21.5, 52.1),
    ("MPE", "simple_tag", "expert"): (151.8, 150.4),
    ("MPE", "simple_tag", "medium-replay"): (69.3, 78.5),
    ("MPE", "simple_tag", "medium"): (111.0, 117.4),
    ("MPE", "simple_tag", "random"): (20.3, 45.9),
    ("MPE", "simple_world", "expert"): (158.3, 160.4),
    ("MPE", "simple_world", "medium-replay"): (52.4, 70.1),
    ("MPE", "simple_world", "medium"): (132.0, 140.3),
    ("MPE", "simple_world", "random"): (4.2, 11.4),
    ("MAMuJoCo", "2ant", "Good"): (2698.0, 2523.0),
    ("MAMuJoCo", "2ant", "Medium"): (1196.0, 1409.0),
    ("MAMuJoCo", "2ant", "Poor"): (681.0, 935.0),
    ("MAMuJoCo", "4ant", "Good"): (2962.0, 3011.0),
    ("MAMuJoCo", "4ant", "Medium"): (1891.0, 2046.0),
    ("MAMuJoCo", "4ant", "Poor"): (1013.0, 1197.0),
    ("SMAC", "3m", "Good"): (18.8, 19.8),
    ("SMAC", "3m", "Medium"): (16.7, 16.9),
    ("SMAC", "3m", "Poor"): (11.7, 12.7),
    ("SMAC", "2s3z", "Good"): (19.8, 20.0),
    ("SMAC", "2s3z", "Medium"): (16.7, 17.2),
    ("SMAC", "2s3z", "Poor"): (10.2, 11.1),
    ("SMAC", "5m_vs_6m", "Good"): (16.9, 18.0),
    ("SMAC", "5m_vs_6m", "Medium"): (16.1, 17.1),
    ("SMAC", "5m_vs_6m", "Poor"): (10.1, 10.6),
    ("SMAC", "8m", "Good"): (19.9, 19.8),
    ("SMAC", "8m", "Medium"): (16.6, 17.7),
    ("SMAC", "8m", "Poor"): (8.8, 9.1),
}


def task_label(domain, task, split):
    return f"{domain} {PRETTY_TASK.get(task, task)}-{PRETTY_TASK.get(split, split)}"


def norm_key(domain, task, split):
    return domain, task.replace("-", "_"), split.replace("-", "_")


def load_ep_json(path):
    with open(path) as f:
        obj = json.load(f)
    vals = np.asarray(obj.get("rews", []), dtype=np.float64)
    wins = np.asarray(obj.get("wins", []), dtype=np.float64)
    return {
        "mean": float(np.nanmean(vals)) if vals.size else np.nan,
        "std": float(np.nanstd(vals)) if vals.size else np.nan,
        "win": float(np.nanmean(wins)) if wins.size else np.nan,
        "n": int(vals.size),
    }


def load_planning_results(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        name = os.path.basename(path)
        if "_policy-only" in name:
            arm = "Reactive"
            stem = name.split("_base_m8_policy-only")[0]
        elif "_WM-plan" in name:
            arm = "Planning"
            stem = name.split("_base_m8_WM-plan")[0]
        else:
            continue
        if stem.startswith("mpe_"):
            rest = stem[len("mpe_") :]
            task, split = None, None
            for mpe_task in ["simple_spread", "simple_tag", "simple_world"]:
                prefix = mpe_task + "-"
                if rest.startswith(prefix):
                    task = mpe_task
                    split = rest[len(prefix) :]
                    break
            if task is None:
                continue
            domain = "MPE"
        elif stem.startswith("mamujoco_"):
            rest = stem[len("mamujoco_") :]
            task, split = rest.rsplit("-", 1)
            domain = "MAMuJoCo"
        elif stem.startswith("smac_"):
            rest = stem[len("smac_") :]
            task, split = rest.rsplit("-", 1)
            domain = "SMAC"
        else:
            continue
        out.setdefault((domain, task, split), {})[arm] = load_ep_json(path)
    return out


def load_horizon_metrics(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or str(row.get("horizon")) != "8":
                continue
            domain = row["domain"]
            task = row["env_name"]
            split = row["split"]
            out[norm_key(domain, task, split)] = {
                "spearman": float(row["spearman"]),
                "pearson": float(row["pearson"]),
                "top1": float(row["pseudo_8way_top1"]),
                "pairwise": float(row["pseudo_8way_pairwise_acc"]),
                "regret": float(row["pseudo_8way_norm_regret"]),
                "state_mse": float(row["state_mse_mean"]),
            }
    return out


def table_records(results, metrics):
    rows = []
    for domain, task, split in TASK_ORDER:
        arms = results.get((domain, task, split), {})
        reported = PAPER_TABLE_RESULTS.get((domain, task, split))
        if reported is not None:
            reactive, planning = reported
            win_delta = (
                arms["Planning"]["win"] - arms["Reactive"]["win"]
                if "Reactive" in arms and "Planning" in arms
                else np.nan
            )
        elif "Reactive" not in arms or "Planning" not in arms:
            continue
        else:
            reactive = arms["Reactive"]["mean"]
            planning = arms["Planning"]["mean"]
            win_delta = arms["Planning"]["win"] - arms["Reactive"]["win"]
        delta = planning - reactive
        rel = 100.0 * delta / max(abs(reactive), 1e-9)
        met = metrics.get(norm_key(domain, task, split), {})
        rows.append({
            "domain": domain,
            "task": task,
            "quality": split,
            "label": task_label(domain, task, split),
            "reactive": reactive,
            "planning": planning,
            "delta": delta,
            "rel_pct": rel,
            "win_delta": win_delta,
            **met,
        })
    return rows


def plot_gain_heatmap(rows, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vals = np.asarray([[r["reactive"], r["planning"], r["delta"], r["rel_pct"]] for r in rows], dtype=float)
    labels = [r["label"] for r in rows]
    cols = ["Reactive", "Planning", "Delta", "Delta %"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.2), gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    raw = vals[:, :2]
    # Row-normalized colors show whether planning improves the same task.
    row_min = np.nanmin(raw, axis=1, keepdims=True)
    row_max = np.nanmax(raw, axis=1, keepdims=True)
    row_norm = (raw - row_min) / np.maximum(row_max - row_min, 1e-9)
    im = ax.imshow(row_norm, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(2), cols[:2], fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.8)
    ax.set_title("(a) Paired return", fontsize=11, pad=8)
    for i in range(len(rows)):
        for j in range(2):
            ax.text(j, i, f"{raw[i,j]:.1f}", ha="center", va="center", fontsize=5.5, color="black")
    ax.tick_params(length=0)
    for y in [12 - 0.5, 18 - 0.5]:
        ax.axhline(y, color="black", lw=1.0)

    ax = axes[1]
    delta_pct = vals[:, 3]
    vmax = np.nanpercentile(np.abs(delta_pct), 90)
    vmax = max(vmax, 10)
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    im2 = ax.imshow(delta_pct[:, None], aspect="auto", cmap="PiYG", norm=norm)
    ax.set_xticks([0], ["Planning gain (%)"], fontsize=10)
    ax.set_yticks(range(len(labels)), [""] * len(labels))
    ax.set_title("(b) Relative gain", fontsize=11, pad=8)
    for i, r in enumerate(rows):
        ax.text(0, i, f"{r['rel_pct']:+.1f}%", ha="center", va="center", fontsize=6.0, color="black")
    ax.tick_params(length=0)
    for y in [12 - 0.5, 18 - 0.5]:
        ax.axhline(y, color="black", lw=1.0)

    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.05, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Planning improves the same frozen policy across task-quality settings", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_line(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return np.polyfit(x[mask], y[mask], 1)


def plot_gain_vs_reliability(rows, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"MPE": "#1f77b4", "MAMuJoCo": "#d62728", "SMAC": "#2ca02c"}
    markers = {"MPE": "o", "MAMuJoCo": "s", "SMAC": "^"}
    xs = [
        ("spearman", "Spearman rank correlation"),
        ("top1", "Pseudo 8-way Top-1"),
        ("regret", "Normalized selection regret"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    y = np.asarray([r["rel_pct"] for r in rows], dtype=float)

    for ax, (key, xlabel) in zip(axes, xs):
        x = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
        for domain in ["MPE", "MAMuJoCo", "SMAC"]:
            idx = [i for i, r in enumerate(rows) if r["domain"] == domain and np.isfinite(x[i]) and np.isfinite(y[i])]
            ax.scatter(x[idx], y[idx], s=38, c=colors[domain], marker=markers[domain], edgecolor="white", linewidth=0.6, label=domain, alpha=0.9)
        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.2, alpha=0.75)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.text(0.05, 0.92, f"r={corr:.2f}", transform=ax.transAxes, fontsize=10)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.tick_params(labelsize=9)
        if key == "regret":
            ax.invert_xaxis()
    axes[0].set_ylabel("Planning gain over reactive (%)", fontsize=10)
    axes[0].legend(frameon=True, fontsize=9, loc="best")
    fig.suptitle("Planning gains are tied to the world model's ability to rank short-horizon futures", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_headroom_gain(rows, out_pdf, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"MPE": "#1f77b4", "MAMuJoCo": "#d62728", "SMAC": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4))
    for ax, domain in zip(axes, ["MPE", "MAMuJoCo", "SMAC"]):
        sub = [r for r in rows if r["domain"] == domain]
        x = np.asarray([r["reactive"] for r in sub], dtype=float)
        y = np.asarray([r["rel_pct"] for r in sub], dtype=float)
        ax.scatter(x, y, s=46, c=colors[domain], edgecolor="white", linewidth=0.6, alpha=0.9)
        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.0, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(domain, fontsize=11)
        ax.set_xlabel("Reactive reported score", fontsize=10)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Planning gain over reactive (%)", fontsize=10)
    fig.suptitle("Headroom diagnostic: gains are largest where the reactive policy is not saturated", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_table(rows, out_csv):
    keys = ["domain", "task", "quality", "reactive", "planning", "delta", "rel_pct", "win_delta", "spearman", "top1", "regret", "state_mse"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--score_metrics", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = load_planning_results(args.results_dir)
    metrics = load_horizon_metrics(args.score_metrics)
    rows = table_records(results, metrics)
    save_table(rows, os.path.join(args.out_dir, "planning_gain_visuals_data.csv"))
    plot_gain_heatmap(rows, os.path.join(args.out_dir, "planning_gain_heatmap.pdf"), os.path.join(args.out_dir, "planning_gain_heatmap.png"))
    plot_gain_vs_reliability(rows, os.path.join(args.out_dir, "planning_gain_vs_reliability.pdf"), os.path.join(args.out_dir, "planning_gain_vs_reliability.png"))
    plot_headroom_gain(rows, os.path.join(args.out_dir, "planning_gain_vs_headroom.pdf"), os.path.join(args.out_dir, "planning_gain_vs_headroom.png"))
    print(f"[SAVED] {args.out_dir} rows={len(rows)}")


if __name__ == "__main__":
    main()
