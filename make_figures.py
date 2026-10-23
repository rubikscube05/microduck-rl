"""
make_figures.py
===============

Generate the diagrams used in the README:

    docs/images/pipeline.png        training pipeline diagram
    docs/images/training_curve.png  evaluation reward and episode length

The training curve needs `logs/eval/evaluations.npz`, which is written by the
`EvalCallback` during training.  If that file does not exist yet, the script
still regenerates the pipeline diagram and skips the curve.

Run:
    ./.venv/bin/python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render to files without needing a display.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# A small shared style so both figures look like they belong together.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#c9d1d9",
        "axes.labelcolor": "#24292f",
        "xtick.color": "#57606a",
        "ytick.color": "#57606a",
        "text.color": "#24292f",
    }
)


def plot_training_curve():
    """Plot reward and episode length from the EvalCallback history."""
    path = ROOT / "logs" / "eval" / "evaluations.npz"
    if not path.exists():
        print(f"[figures] {path} not found; skipping training curve")
        return

    data = np.load(path)
    steps = data["timesteps"] / 1e6
    reward = data["results"].mean(axis=1)
    reward_std = data["results"].std(axis=1)
    length = data["ep_lengths"].mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#ffffff")

    ax1.fill_between(
        steps, reward - reward_std, reward + reward_std,
        color="#2da44e", alpha=0.18, linewidth=0,
    )
    ax1.plot(steps, reward, color="#1a7f37", linewidth=2.4)
    ax1.axhline(0.9 * 1000, color="#8c959f", linestyle="--", linewidth=1.2)
    ax1.text(
        steps[2], 0.9 * 1000 + 90, "standing still (~900)",
        color="#57606a", fontsize=9,
    )
    ax1.set_ylabel("Episode reward", fontsize=11)
    ax1.set_title(
        "PPO learning curve: Microduck learns to walk",
        fontsize=15, weight="bold", loc="left", pad=12,
    )
    ax1.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(steps, length, color="#0969da", linewidth=2.2)
    ax2.axhline(1000, color="#8c959f", linestyle="--", linewidth=1.2)
    ax2.text(steps[2], 965, "full episode (1000 steps)", color="#57606a", fontsize=9)
    ax2.set_ylim(0, 1080)
    ax2.set_ylabel("Episode length", fontsize=11)
    ax2.set_xlabel("Environment steps (millions)", fontsize=11)
    ax2.grid(axis="y", color="#d8dee4", linewidth=0.8, alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.text(
        0.08, 0.02,
        "Reward strengthened at 2M steps; fine-tuned for 3M more.",
        fontsize=8.5, color="#57606a",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)
    fig.savefig(OUT / "training_curve.png", dpi=160, facecolor="#ffffff")
    plt.close(fig)
    print("[figures] wrote docs/images/training_curve.png")


def plot_pipeline():
    """Draw the URDF -> MuJoCo -> Gymnasium -> PPO data flow."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    boxes = [
        (0.25, 0.85, 2.5, 1.5, "#ddf4ff", "#0969da",
         "Microduck URDF", "15 joints, 25 cm tall"),
        (3.15, 0.85, 2.5, 1.5, "#dafbe1", "#1a7f37",
         "MuJoCo model", "physics at 500 Hz"),
        (6.05, 0.85, 2.5, 1.5, "#fff8c5", "#9a6700",
         "Gymnasium env", "41-d obs / 10-d actions"),
        (8.95, 0.85, 2.8, 1.5, "#fbefff", "#8250df",
         "PPO policy", "2 x 256 MLP actor-critic"),
    ]

    for x, y, w, h, fill, edge, title, subtitle in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=1.6, edgecolor=edge, facecolor=fill,
            )
        )
        ax.text(
            x + w / 2, y + h * 0.62, title,
            ha="center", va="center", fontsize=12,
            weight="bold", color="#24292f",
        )
        ax.text(
            x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=9, color="#57606a",
        )

    for x0, x1 in [(2.75, 3.15), (5.65, 6.05), (8.55, 8.95)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.6), (x1, 1.6),
                arrowstyle="-|>", mutation_scale=16,
                linewidth=1.6, color="#8c959f",
            )
        )

    # Feedback loop: actions, rewards, and the next observations.
    ax.add_patch(
        FancyArrowPatch(
            (10.35, 0.82), (10.35, 0.38),
            arrowstyle="-", linewidth=1.5, color="#8c959f",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (10.35, 0.38), (7.30, 0.38),
            arrowstyle="-", linewidth=1.5, color="#8c959f",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (7.30, 0.38), (7.30, 0.82),
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.5, color="#8c959f",
        )
    )
    ax.text(
        8.8, 0.50, "actions / rewards / next observations",
        ha="center", va="center", fontsize=9, color="#57606a",
    )
    ax.set_title(
        "Training pipeline", fontsize=15, weight="bold",
        loc="left", x=0.02, pad=10,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "pipeline.png", dpi=160,
        facecolor="#ffffff", bbox_inches="tight",
    )
    plt.close(fig)
    print("[figures] wrote docs/images/pipeline.png")


if __name__ == "__main__":
    plot_training_curve()
    plot_pipeline()
