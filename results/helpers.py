import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_cvar(pnl_array, alpha=0.95):
    """Loss CVaR at confidence alpha: lower is better (loss = -P&L)."""
    pnl_array = np.asarray(pnl_array)
    if (
        pnl_array.ndim != 1
        or not pnl_array.size
        or not 0 < alpha < 1
        or not np.isfinite(pnl_array).all()
    ):
        raise ValueError("CVaR requires finite P&L and a confidence level in (0, 1)")
    cutoff = max(1, int(round((1.0 - alpha) * len(pnl_array))))
    return -np.partition(pnl_array, cutoff - 1)[:cutoff].mean()


def get_available_cost_rates(directory="."):
    """Cost rates with exported neural-network P&L, sorted numerically."""
    files = Path(directory).glob("pnl/final_pnl_nn_cost_rate_*.csv")
    rates = {path.stem.removeprefix("final_pnl_nn_cost_rate_") for path in files}
    return sorted(rates, key=float)


def plot_pnl_comparisons(
    rates, input_dir=".", output_dir=".", baseline: Literal["Ref", "BS"] = "BS"
):
    if baseline not in ("Ref", "BS"):
        raise ValueError("Baseline must be 'Ref' or 'BS'")
    if not rates:
        raise ValueError("At least one cost rate is required")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    baseline_lw = baseline.lower()

    num_plots = len(rates)
    cols = 2
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 6 * rows))
    axes = np.asarray(axes).reshape(-1)

    for i, rate in enumerate(rates):
        rate = f"{float(rate):g}"
        ax = axes[i]
        nn_file = os.path.join(input_dir, "pnl", f"final_pnl_nn_cost_rate_{rate}.csv")
        base_file = os.path.join(input_dir, "pnl", f"final_pnl_{baseline_lw}_cost_rate_{rate}.csv")

        pnl_nn = pd.read_csv(nn_file)["PNL"].values
        pnl_baseline = pd.read_csv(base_file)["PNL"].values

        cvar_nn = calculate_cvar(pnl_nn)
        cvar_baseline = calculate_cvar(pnl_baseline)

        base_label = "Black-Scholes" if baseline == "BS" else "Reference (frictionless-trained)"
        sns.histplot(
            pnl_baseline, bins=100, color="gray", alpha=0.5,
            label=f"{base_label} (CVaR: {cvar_baseline:.2f})", stat="density", ax=ax,
        )
        sns.histplot(
            pnl_nn, bins=100, color="royalblue", alpha=0.6,
            label=f"Neural network (CVaR: {cvar_nn:.2f})", stat="density", ax=ax,
        )

        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(f"Neural network vs {base_label}: net P&L (Cost Rate: {rate})", fontweight='bold')
        ax.set_xlabel("Portfolio P&L at Maturity")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    out_name = os.path.join(output_dir, f"plot_pnl_comparison_vs_{baseline}.png")
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.show()
    plt.close(fig)


def plot_loss_history(rates, input_dir=".", output_dir="."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    rates_ft = [f"{float(r):g}" for r in rates if float(r) != 0.0]

    colors_ft = sns.color_palette("rocket", len(rates_ft))

    ax_left = axes[0]
    loss_file = os.path.join(input_dir, "losses", "loss_history_cost_rate_0.csv")

    df = pd.read_csv(loss_file)
    sns.lineplot(data=df, x="Epoch", y="Loss", ax=ax_left,
                    color="darkorange", linewidth=2.5, label="Reference (frictionless-trained)")

    ax_left.set_title("Pre-Training: Frictionless Market", fontweight='bold')
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("95% loss CVaR (lower is better)")
    ax_left.legend()

    ax_right = axes[1]
    for idx, rate in enumerate(rates_ft):
        loss_file = os.path.join(input_dir, "losses", f"loss_history_cost_rate_{rate}.csv")
        df = pd.read_csv(loss_file)
        df["Loss_norm"] = df["Loss"] / df.iloc[0]["Loss"]

        sns.lineplot(data=df, x="Epoch", y="Loss_norm", ax=ax_right,
                     color=colors_ft[idx], linewidth=2, label=f"Cost = {rate}")

    ax_right.set_title("Fine-Tuning: Market Frictions", fontweight='bold')
    ax_right.set_xlabel("Epoch")
    ax_right.set_ylabel("Loss CVaR / first-epoch loss CVaR")
    if rates_ft:
        ax_right.legend(title="Transaction Cost Rate")

    out_name = os.path.join(output_dir, "plot_split_loss_history.png")
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)
    plt.show()
    plt.close(fig)


def plot_turnover_vs_cvar(rates, input_dir=".", output_dir=".", alpha=0.95):
    """Compare policies at matching costs; turnover includes entry and liquidation."""
    rates = sorted(set(float(rate) for rate in rates))
    if not rates:
        raise ValueError("At least one cost rate is required")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    for policy, color, marker, style, label in [
        ("Ref", "darkgray", "x", "--", "Reference (frictionless-trained)"),
        ("NN", "crimson", "o", "-", "Cost-aware NN (fine-tuned)")
    ]:
        turnover, cvar = [], []
        for rate in rates:
            pnl = pd.read_csv(input_dir / "pnl" /
                              f"final_pnl_{policy.lower()}_cost_rate_{rate:g}.csv")["PNL"]
            metrics = pd.read_csv(input_dir / "metrics" /
                                  f"final_path_metrics_cost_rate_{rate:g}.csv")
            turnover.append(metrics[f"Turnover_{policy}"].mean())
            cvar.append(calculate_cvar(pnl, alpha))
        ax.plot(turnover, cvar, marker=marker, linestyle=style, color=color,
                linewidth=2, markersize=8, label=label)
        for rate, volume, risk in zip(rates, turnover, cvar):
            # Low-cost points lie close to the common starting point.
            if 0 < rate < 0.005 or (rate == 0 and policy == "Ref"):
                continue
            ax.annotate(f"c={100 * rate:g}%", (volume, risk),
                        textcoords="offset points",
                        xytext=(8, 8 if policy == "NN" else -12),
                        fontsize=9, color=color)
    ax.set_title("Cost-aware NN vs reference: turnover and loss CVaR", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel(r"Mean Portfolio Turnover ($\sum |\Delta_t - \Delta_{t-1}|$)")
    ax.set_ylabel(f"{alpha * 100:g}% loss CVaR (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "turnover_vs_cvar_frontier.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cvar_gain_vs_reference(rates, input_dir=".", output_dir="."):
    """Relative loss-CVaR reduction with approximate paired antithetic 95% CIs.

    CIs are pointwise, conditional on checkpoints, not across training seeds.
    Antithetic partners must be stored in the first and second batch halves.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rate in sorted(set(float(r) for r in rates)):
        rate = f"{rate:g}"
        losses_nn = -pd.read_csv(input_dir / "pnl" / f"final_pnl_nn_cost_rate_{rate}.csv")["PNL"].to_numpy()
        losses_ref = -pd.read_csv(input_dir / "pnl" / f"final_pnl_ref_cost_rate_{rate}.csv")["PNL"].to_numpy()
        n = len(losses_nn)
        if n != len(losses_ref) or n % 2 or n < 40:
            raise ValueError(f"Unaligned or invalid antithetic batch at c={rate}")
        if not np.isfinite(losses_nn).all() or not np.isfinite(losses_ref).all():
            raise ValueError(f"Non-finite P&L at c={rate}")
        tail_n = int(round(.05 * n))
        es_nn = np.partition(losses_nn, -tail_n)[-tail_n:].mean()
        es_ref = np.partition(losses_ref, -tail_n)[-tail_n:].mean()
        if es_ref <= 0:
            raise ValueError("Relative ES reduction requires positive reference ES")
        # ES influence scores, paired both across policies and antithetic paths.
        q_nn, q_ref = np.quantile(losses_nn, .95), np.quantile(losses_ref, .95)
        score_nn = q_nn + np.maximum(losses_nn - q_nn, 0) / .05
        score_ref = q_ref + np.maximum(losses_ref - q_ref, 0) / .05
        gain = 100 * (1 - es_nn / es_ref)
        influence = -100 / es_ref * score_nn + 100 * es_nn / es_ref**2 * score_ref
        paired = (influence[:n//2] + influence[n//2:]) / 2
        se = paired.std(ddof=1) / np.sqrt(n//2)
        rows.append([float(rate), n, es_ref, es_nn, gain, gain-1.96*se, gain+1.96*se])

    summary = pd.DataFrame(rows, columns=["Cost_Rate", "Paths", "ES95_Ref", "ES95_NN",
                                          "Reduction_Percent", "CI95_Lower", "CI95_Upper"])
    if summary.empty:
        raise FileNotFoundError("No final P&L CSVs found")
    summary.to_csv(output_dir / "cvar_gain_vs_reference.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = summary.Cost_Rate.to_numpy() * 100
    gain = summary.Reduction_Percent.to_numpy()
    ax.errorbar(x, gain, yerr=np.vstack([gain-summary.CI95_Lower, summary.CI95_Upper-gain]),
                fmt="o-", color="crimson", capsize=5, linewidth=1.8,
                label="Cost-aware NN (fine-tuned) vs reference")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Incremental tail-risk benefit of cost-aware training",
           xlabel="Proportional transaction cost (%)",
           ylabel="Reduction in 95% loss CVaR (%)")
    ax.legend(loc="upper left")
    for index in [1, 5, len(summary)-1]:
        if index < len(summary):
            ax.annotate(f"{gain[index]:.2f}%", (x[index], gain[index]),
                        xytext=(0, 10), textcoords="offset points", ha="center")
    ax.margins(x=.06, y=.15)
    fig.text(.5, .025, "95% pointwise Monte Carlo CIs: paired policies and antithetic pairs."
             " Conditional on fixed checkpoints; not training-seed uncertainty.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output_dir / "cvar_gain_vs_reference.png", dpi=300)
    plt.show()
    plt.close(fig)

    return summary


def plot_transaction_cost_distributions(rates=(0.005, 0.02), input_dir=".", output_dir="."):
    """Cumulative fee distributions; x-axis zooms exclude only the extreme tail."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rates = sorted(set(float(r) for r in rates))
    if not rates:
        raise ValueError("At least one cost rate is required")
    fig, axes = plt.subplots(1, len(rates), figsize=(6*len(rates), 5), sharey=True, squeeze=False)
    axes = axes[0]
    cost_rows = []
    for ax, rate in zip(axes, rates):
        rate = f"{rate:g}"
        frame = pd.read_csv(input_dir / "metrics" / f"final_path_metrics_cost_rate_{rate}.csv")
        if not np.array_equal(frame.Path.to_numpy(), np.arange(len(frame))):
            raise ValueError("Unexpected path IDs")
        pooled = frame[["Total_Cost_Ref", "Total_Cost_NN"]].to_numpy()
        if not np.isfinite(pooled).all() or np.any(pooled < 0):
            raise ValueError("Invalid cumulative fees")
        right = np.quantile(pooled, .995)
        for name, color, style in [("Ref", "gray", "--"), ("NN", "royalblue", "-")]:
            values = frame[f"Total_Cost_{name}"].to_numpy()
            # Quantile sampling keeps the empirical distribution compact.
            probabilities = np.linspace(0, 1, 2001)
            quantiles = np.quantile(values, probabilities)
            label = "Reference (frictionless-trained)" if name == "Ref" else "Cost-aware NN (fine-tuned)"
            ax.plot(quantiles, probabilities * 100, color=color, linestyle=style,
                    linewidth=2, label=f"{label} (mean {values.mean():.3f})")
            cost_rows.append([float(rate), name, len(values), values.mean(),
                              *np.quantile(values, [.05, .5, .95, .995])])
        ax.set_xlim(0, right * 1.03)
        ax.set_ylim(0, 101)
        ax.set(title=f"Transaction cost = {float(rate)*100:g}%",
               xlabel="Cumulative transaction fees (P&L units)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Cumulative probability (%)")
    fig.suptitle("Cost-aware NN vs reference: cumulative execution costs", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output_dir / "transaction_cost_distributions.png", dpi=300)
    plt.show()
    plt.close(fig)
    pd.DataFrame(cost_rows, columns=["Cost_Rate", "Policy", "Paths", "Mean", "P05",
                                    "Median", "P95", "P995"]).to_csv(
        output_dir / "transaction_cost_distribution_summary.csv", index=False)


def calculate_delta_stability(cost_rates, input_dir=".", small_trade_threshold=1e-4):
    """Per-path statistics, excluding entry and liquidation.

    Reversals count changes of sign between trades above the given threshold.
    These statistics describe the exported sample, not the full test batch.
    """
    if np.isscalar(cost_rates):
        cost_rates = [cost_rates]
    if small_trade_threshold <= 0:
        raise ValueError("The small-trade threshold must be positive")
    rows = []
    for rate in sorted(set(float(r) for r in cost_rates)):
        filename = os.path.join(input_dir, "deltas", f"sample_path_deltas_cost_rate_{rate:g}.csv")
        df = pd.read_csv(filename).sort_values(["Path", "TimeStep"])
        if df.empty or not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"Invalid trajectory data at c={rate:g}")
        for path, group in df.groupby("Path"):
            if len(group) < 3 or not np.array_equal(group["TimeStep"], np.arange(len(group))):
                raise ValueError(f"Incomplete trajectory {path}")
            for policy in ["BS", "Ref", "NN"]:
                # Positions at t=0,...,N-1; diff gives trades at t=1,...,N-1.
                changes = np.diff(group[f"Delta_{policy}"].to_numpy()[:-1])
                absolute = np.abs(changes)
                active_changes = changes[absolute >= small_trade_threshold]
                rows.append({
                    "Cost_Rate": rate,
                    "Path": path,
                    "Policy": policy,
                    "Intermediate_Turnover": absolute.sum(),
                    "Mean_Abs_Change": absolute.mean(),
                    "Squared_Changes": np.sum(changes**2),
                    "Max_Abs_Change": absolute.max(),
                    "Reversals": np.sum(active_changes[1:]*active_changes[:-1] < 0),
                    "Small_Adjustment_Percent": 100*np.mean(absolute < small_trade_threshold)
                })
    if not rows:
        raise ValueError("At least one cost rate is required")
    return pd.DataFrame(rows)


def plot_delta_stability(cost_rates, input_dir=".", output_dir=".", small_trade_threshold=1e-4):
    """Compare sample-mean stability metrics across transaction cost rates."""
    stats = calculate_delta_stability(cost_rates, input_dir, small_trade_threshold)
    metrics = ["Intermediate_Turnover", "Squared_Changes"]
    titles = ["Mean turnover", "Sum of squared position changes"]
    units = ["Shares", "Shares squared"]
    summary = stats.groupby(["Cost_Rate", "Policy"])[metrics +
        ["Max_Abs_Change", "Reversals", "Small_Adjustment_Percent"]].mean()
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, "delta_stability_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, metric, title, unit in zip(axes, metrics, titles, units):
        for policy, label, color, marker, style in [
            ("Ref", "Reference (frictionless-trained)", "darkorange", "s", "--"),
            ("NN", "Cost-aware NN (fine-tuned)", "royalblue", "o", "-")]:
            values = summary.xs(policy, level="Policy")[metric]
            ax.plot(values.index.to_numpy()*100, values, color=color,
                    marker=marker, linestyle=style, label=label)
        ax.set(title=title, xlabel="Proportional transaction cost (%)", ylabel=unit)
        ax.legend()
        if (0.01, "NN") in summary.index:
            reference = summary.loc[(0.01, "Ref"), metric]
            nn = summary.loc[(0.01, "NN"), metric]
            reduction = 100*(1-nn/reference)
            ax.annotate(f"{reduction:.1f}% reduction at c=1%", (1, nn),
                        xytext=(25, 35), textcoords="offset points",
                        arrowprops={"arrowstyle": "->", "color": "royalblue"},
                        fontsize=10)
    fig.suptitle("Cost-aware NN vs reference: delta stability")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(os.path.join(output_dir, "delta_stability_by_cost.png"), dpi=300)
    plt.show()
    plt.close(fig)

    return stats


def plot_comprehensive_metrics_dashboard(rates, input_dir=".", output_dir="."):
    """Mean net P&L and RMSE over all test paths, without single-path metrics."""
    valid_rates = sorted(set(float(r) for r in rates))
    if not valid_rates:
        raise ValueError("At least one cost rate is required")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.asarray(valid_rates) * 100
    for policy, color, marker, style, label in [
        ("ref", "gray", "x", "--", "Reference (frictionless-trained)"),
        ("nn", "royalblue", "o", "-", "Cost-aware NN (fine-tuned)")]:
        means, rmses = [], []
        for rate in valid_rates:
            pnl = pd.read_csv(os.path.join(input_dir, "pnl",
                f"final_pnl_{policy}_cost_rate_{rate:g}.csv"))["PNL"].to_numpy()
            if not len(pnl) or not np.isfinite(pnl).all():
                raise ValueError(f"Invalid final P&L at c={rate:g}")
            means.append(pnl.mean())
            rmses.append(np.sqrt(np.mean(pnl**2)))
        axes[0].plot(x, means, color=color, marker=marker, linestyle=style, label=label)
        axes[1].plot(x, rmses, color=color, marker=marker, linestyle=style, label=label)
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[0].set(title="Mean net hedging P&L", ylabel="Mean P&L (P&L units)")
    axes[1].set(title="Root mean squared hedging error", ylabel="RMSE (P&L units)")
    for ax in axes:
        ax.set_xlabel("Proportional transaction cost (%)")
        ax.legend()
    fig.suptitle("Cost-aware NN vs reference: mean net P&L and RMSE")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "comprehensive_metrics_dashboard.png"), dpi=300)
    plt.show()
    plt.close(fig)
