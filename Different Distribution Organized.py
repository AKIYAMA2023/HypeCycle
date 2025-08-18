import numpy as np
import matplotlib.pyplot as plt

from Distrbituion import Model  # Ensure Distrbituion.py is in the same folder

###############################################################################
# 3 × 4 GRID PLOT WITH GROUP‑SPECIFIC Y‑AXES                                   #
# ─────────────────────────────────────────────────────────────────────────── #
# • Every panel shows y‑tick numbers (so ranges are readable).               #
# • Y‑axis *labels* appear **only** on the first column (col 0).             #
###############################################################################

def compute_theoretical_pdf(x: np.ndarray, dist_type: str, params: dict) -> np.ndarray:
    if dist_type == "normal":
        mean, std = params.get("mean", 0), params.get("std", 1)
        return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    if dist_type == "uniform":
        low, high = params.get("low", -1), params.get("high", 1)
        return np.where((x >= low) & (x <= high), 1 / (high - low), 0)
    if dist_type == "bimodal":
        m1, m2 = params.get("mean1", -0.5), params.get("mean2", 0.5)
        s1, s2 = params.get("std1", 0.2), params.get("std2", 0.2)
        w = params.get("weight", 0.5)
        pdf1 = 1 / (s1 * np.sqrt(2 * np.pi)) * np.exp(-((x - m1) ** 2) / (2 * s1 ** 2))
        pdf2 = 1 / (s2 * np.sqrt(2 * np.pi)) * np.exp(-((x - m2) ** 2) / (2 * s2 ** 2))
        return w * pdf1 + (1 - w) * pdf2
    raise ValueError("Unknown distribution type")


def run_distribution_grid(
    n_runs: int = 100,
    ticks: int = 400,
    *,
    agent_count: int = 999,
    learning_rate: float = 0.3,
    confidence_threshold: float = 0.5,
    save_path: str | None = None,
):
    """Run simulations and generate a 3 × 4 grid with group‑specific y‑scales."""

    distributions = {
        "Uniform": {
            "type": "uniform",
            "params": {"low": -1.5, "high": 1.5},
        },
        "Bimodal Wide": {
            "type": "bimodal",
            "params": {"mean1": -0.8, "mean2": 0.8, "std1": 0.3, "std2": 0.3, "weight": 0.5},
        },
        "Bimodal Close": {
            "type": "bimodal",
            "params": {"mean1": -0.5, "mean2": 0.5, "std1": 0.3, "std2": 0.3, "weight": 0.5},
        },
        "Normal": {
            "type": "normal",
            "params": {"mean": 0, "std": 0.5},
        },
    }
    order = list(distributions.keys())
    left_dists, right_dists = order[:2], order[2:]

    t = np.arange(ticks + 1)
    sim_results = {}

    for name, cfg in distributions.items():
        sum_runs, adv_runs = [], []
        for _ in range(n_runs):
            model = Model(
                agent_count=agent_count,
                learning_rate=learning_rate,
                confidence_threshold=confidence_threshold,
                distribution_type=cfg["type"],
                dist_params=cfg["params"],
            )
            model.run(ticks)
            # sum_opinion should be sum of expressed_opinion for aware agents
            sum_runs.append(np.asarray(model.sum_opinion))
            # aware_positive_count should be count of aware agents with expressed_opinion > 0.5
            adv_runs.append(np.asarray(model.aware_positive_count))

        sum_runs = np.vstack(sum_runs)
        adv_runs = np.vstack(adv_runs)
        sim_results[name] = {
            "mean_sum": sum_runs.mean(axis=0),
            "ci_sum": 1.96 * sum_runs.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.zeros(ticks + 1),
            "mean_adv": adv_runs.mean(axis=0),
            "ci_adv": 1.96 * adv_runs.std(axis=0, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else np.zeros(ticks + 1),
        }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=False)
    x_pdf = np.linspace(-2, 2, 1000)

    for col, name in enumerate(order):
        cfg = distributions[name]
        pdf_vals = compute_theoretical_pdf(x_pdf, cfg["type"], cfg["params"])
        axes[0, col].plot(x_pdf, pdf_vals, color="gray", lw=2)
        axes[0, col].set_title(name, fontsize=14)
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(t, sim_results[name]["mean_sum"], color="blue", lw=2)
        axes[1, col].fill_between(
            t,
            sim_results[name]["mean_sum"] - sim_results[name]["ci_sum"],
            sim_results[name]["mean_sum"] + sim_results[name]["ci_sum"],
            color="blue",
            alpha=0.1,
        )
        axes[1, col].grid(alpha=0.3)

        axes[2, col].plot(t, sim_results[name]["mean_adv"], color="green", lw=2)
        axes[2, col].fill_between(
            t,
            sim_results[name]["mean_adv"] - sim_results[name]["ci_adv"],
            sim_results[name]["mean_adv"] + sim_results[name]["ci_adv"],
            color="green",
            alpha=0.1,
        )
        axes[2, col].set_xlabel("Time Step (t)")
        axes[2, col].grid(alpha=0.3)

    # Add x-axis label for the first row
    for ax in axes[0, :]:
        ax.set_xlabel("Latent Opinion", fontsize=12)

    # Add labels (a, b, ...) outside the top-left corner of each plot
    labels = list("abcdefghijkl")
    for col, ax in enumerate(axes.flat):
        ax.text(-0.05, 1.02, labels[col], transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="center")

    # ──────────────────────────────────────────────────────────────────
    # AXIS RANGE TUNING – LEFT vs RIGHT GROUPS PER ROW
    # ──────────────────────────────────────────────────────────────────
    def set_group_ylim(row: int, dists: list[str], cols: list[int], margin: float = 0.1):
        if row == 0:
            ymax = max(
                compute_theoretical_pdf(x_pdf, distributions[d]["type"], distributions[d]["params"]).max()
                for d in dists
            ) * (1 + margin)
            ymin = 0
        elif row == 1:
            vals = [sim_results[d]["mean_sum"] for d in dists]
            ymin, ymax = min(v.min() for v in vals), max(v.max() for v in vals)
            span = ymax - ymin
            ymin -= margin * span
            ymax += margin * span
        else:
            ymax = max(sim_results[d]["mean_adv"].max() for d in dists) * (1 + margin)
            ymin = 0
        for c in cols:
            axes[row, c].set_ylim(ymin, ymax)

    left_cols, right_cols = [0, 1], [2, 3]
    for r in range(3):
        set_group_ylim(r, left_dists, left_cols)
        set_group_ylim(r, right_dists, right_cols)

    # ──────────────────────────────────────────────────────────────────
    # Y‑LABELS: keep only on first column
    # ──────────────────────────────────────────────────────────────────
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Sum of Expressed Opinions")
    axes[2, 0].set_ylabel("Number of Advocates")
    for r in range(3):
        for c in range(1, 4):
            axes[r, c].set_ylabel("")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return sim_results


if __name__ == "__main__":
    run_distribution_grid(n_runs=100, ticks=300,save_path="distribution_grid_plot.png")
