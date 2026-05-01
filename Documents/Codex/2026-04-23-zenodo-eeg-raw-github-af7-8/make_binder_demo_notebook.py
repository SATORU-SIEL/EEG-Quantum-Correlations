from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "binder_demo_5figures_precomputed.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = [
    md_cell(
        """
        # Binder Demo: Five Structure-First Figures from Precomputed CSV Files

        This notebook is a lightweight Binder-friendly demo.
        It does **not** require the embedded `experiments` object and does **not** recompute EEG features,
        topology, or null simulations from raw data.

        Instead, it redraws the public figures directly from precomputed CSV artifacts already bundled in this repository.
        """
    ),
    md_cell(
        """
        ## What this Binder demo supports

        ### Full five-figure support

        The full five-figure pipeline is available for the published primary frame:

        - `4ch / 60corr / 26task`

        ### Comparison-only support

        The following frames are also available through precomputed summaries and can appear in Figure 5:

        - `4ch / 60corr / 30task`
        - `4ch / 20corr / 26task`
        - `14ch / 20corr / 30task`

        These comparison frames do **not** all have matching precomputed null or sliding-window bundles, so Figure 1, Figure 3, and Figure 4 remain tied to the published primary frame.
        """
    ),
    code_cell(
        """
        # Demo configuration

        %matplotlib inline

        from pathlib import Path

        REPRO_CSV_DIR = Path("Repro_CSV")
        REPRO_FIG_DIR = Path("Repro_Figure")
        REPRO_CSV_DIR.mkdir(parents=True, exist_ok=True)
        REPRO_FIG_DIR.mkdir(parents=True, exist_ok=True)

        RESULTS_DIR = Path("results")

        PRIMARY_FRAME = "4ch_60corr_26task"

        FRAME_COMPARISON = [
            "4ch_60corr_26task",
            "4ch_20corr_26task",
            "14ch_20corr_30task",
        ]

        print("CSV output directory:", REPRO_CSV_DIR.resolve())
        print("Figure output directory:", REPRO_FIG_DIR.resolve())
        print("Primary frame:", PRIMARY_FRAME)
        print("Frame comparison:", FRAME_COMPARISON)
        """
    ),
    md_cell(
        """
        ## Frame guide

        The available frame labels are:

        - `4ch_60corr_26task`
        - `4ch_60corr_30task`
        - `4ch_20corr_26task`
        - `14ch_20corr_30task`

        The first frame is the full published structure-first frame.
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import math

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from matplotlib.patches import Ellipse

        plt.style.use("default")


        FRAME_REGISTRY = {
            "4ch_60corr_26task": {
                "label": "4ch / 60corr / 26task",
                "session_summary": RESULTS_DIR / "task_topology_link_4ch60corr26_single_figure" / "session_topology_signature_values.csv",
                "top_rest": RESULTS_DIR / "structure_first_figures" / "Figure2_top_rest_points.csv",
                "sliding_agg": RESULTS_DIR / "peak_task_sliding_topology_4ch60corr26" / "bestpair_peak_sliding_profile_aggregate.csv",
                "null_observed": RESULTS_DIR / "task_topology_link_4ch60corr26_null_signature_fast" / "observed17v9_vs_null_summary.csv",
                "null_distribution": RESULTS_DIR / "task_topology_link_4ch60corr26_null_signature_fast" / "null_group_median_summaries.csv",
                "best_summary": RESULTS_DIR / "eqcorr_session_wise_best_60correlations_cutback4_two_stage_within_session_summary.csv",
                "null_counts": RESULTS_DIR / "fixed_condition_false_matching_and_fake_quantum_nulls.csv",
                "null_combo": "fixed_4ch_AF3_F7_FC5_FC6",
            },
            "4ch_60corr_30task": {
                "label": "4ch / 60corr / 30task",
                "session_summary": RESULTS_DIR / "task_topology_link_4ch60corr30" / "session_topology_summary.csv",
                "top_rest": None,
                "sliding_agg": None,
                "null_observed": None,
                "null_distribution": None,
                "best_summary": RESULTS_DIR / "eqcorr_session_wise_best_60correlations_two_stage_within_session_summary.csv",
                "null_counts": None,
                "null_combo": None,
            },
            "4ch_20corr_26task": {
                "label": "4ch / 20corr / 26task",
                "session_summary": RESULTS_DIR / "task_topology_link_4ch20corr26" / "session_topology_summary.csv",
                "top_rest": None,
                "sliding_agg": None,
                "null_observed": None,
                "null_distribution": None,
                "best_summary": None,
                "null_counts": None,
                "null_combo": None,
            },
            "14ch_20corr_30task": {
                "label": "14ch / 20corr / 30task",
                "session_summary": RESULTS_DIR / "task_topology_link_all14_20corr30" / "session_topology_summary.csv",
                "top_rest": None,
                "sliding_agg": None,
                "null_observed": None,
                "null_distribution": None,
                "best_summary": None,
                "null_counts": None,
                "null_combo": None,
            },
        }


        def zscore(x):
            x = np.asarray(x, float)
            m = np.isfinite(x)
            out = np.full_like(x, np.nan)
            if m.sum() < 2:
                return out
            mu = np.nanmean(x[m])
            sd = np.nanstd(x[m])
            if sd <= 1e-12:
                return out
            out[m] = (x[m] - mu) / sd
            return out


        def moving_average_centered(x, w=3):
            return pd.Series(np.asarray(x, float)).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


        def bootstrap_median_diff(sig_vals, non_vals, n_boot=4000, seed=20260501):
            sig_vals = np.asarray(sig_vals, float)
            non_vals = np.asarray(non_vals, float)
            sig_vals = sig_vals[np.isfinite(sig_vals)]
            non_vals = non_vals[np.isfinite(non_vals)]
            if len(sig_vals) < 2 or len(non_vals) < 2:
                return np.nan, np.nan, np.nan
            rng = np.random.default_rng(seed)
            diffs = np.empty(n_boot, dtype=float)
            for i in range(n_boot):
                s = rng.choice(sig_vals, size=len(sig_vals), replace=True)
                n = rng.choice(non_vals, size=len(non_vals), replace=True)
                diffs[i] = float(np.nanmedian(s) - np.nanmedian(n))
            return float(np.nanmedian(diffs)), float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


        def add_cov_ellipse(ax, x, y, color, n_std=1.5, alpha=0.14, lw=1.6):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3:
                return
            cov = np.cov(x[m], y[m])
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
            ellipse = Ellipse(
                xy=(float(np.mean(x[m])), float(np.mean(y[m]))),
                width=float(width),
                height=float(height),
                angle=float(angle),
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=lw,
            )
            ax.add_patch(ellipse)


        def save_figure(fig, stem, show=True):
            png = REPRO_FIG_DIR / f"{stem}.png"
            pdf = REPRO_FIG_DIR / f"{stem}.pdf"
            fig.savefig(png, dpi=300, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            print("Saved figure:", png)
            print("Saved figure:", pdf)
            if show:
                display(fig)
            plt.close(fig)


        def load_frame_session_summary(frame_key):
            reg = FRAME_REGISTRY[frame_key]
            df = pd.read_csv(reg["session_summary"])
            if "best_abs_r" not in df.columns and "best_r" in df.columns:
                df["best_abs_r"] = np.abs(pd.to_numeric(df["best_r"], errors="coerce"))
            if "top_lambda2" not in df.columns and "top_lambda2_mean" in df.columns:
                df["top_lambda2"] = df["top_lambda2_mean"]
                df["rest_lambda2"] = df["rest_lambda2_mean"]
                df["top_aspl"] = df["top_aspl_mean"]
                df["rest_aspl"] = df["rest_aspl_mean"]
                df["top_h1_zero"] = df["top_h1_zero_mean"]
                df["rest_h1_zero"] = df["rest_h1_zero_mean"]
            return df
        """
    ),
    md_cell(
        """
        ## Figure 1

        Figure 1 is available only for the published primary frame because the bundled null-count summary
        was computed for that frame.
        """
    ),
    code_cell(
        """
        if PRIMARY_FRAME != "4ch_60corr_26task":
            print("Figure 1 is only available for PRIMARY_FRAME = '4ch_60corr_26task'.")
        else:
            reg = FRAME_REGISTRY[PRIMARY_FRAME]
            best_summary = pd.read_csv(reg["best_summary"])
            null_df = pd.read_csv(reg["null_counts"])
            combo = reg["null_combo"]

            true_n = int((best_summary["n_stage2_fdr05"] > 0).sum())
            false_vals = null_df[(null_df["combo"] == combo) & (null_df["condition"] == "mismatched_session")]["surviving_sessions"].to_numpy(float)
            fake_vals = null_df[(null_df["combo"] == combo) & (null_df["condition"].str.contains("random_fake_quantum"))]["surviving_sessions"].to_numpy(float)

            plot_df = pd.DataFrame([
                {"label": "True", "mean_sig_n": float(true_n), "sd_sig_n": 0.0},
                {"label": "False session", "mean_sig_n": float(np.mean(false_vals)), "sd_sig_n": float(np.std(false_vals, ddof=1))},
                {"label": "Random quantum", "mean_sig_n": float(np.mean(fake_vals)), "sd_sig_n": float(np.std(fake_vals, ddof=1))},
            ])

            fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
            x = np.arange(len(plot_df))
            ax.bar(x, plot_df["mean_sig_n"], yerr=plot_df["sd_sig_n"], color=["#111827", "#d97706", "#2563eb"], alpha=0.88, capsize=5)
            for i, row in plot_df.iterrows():
                txt = f"{int(row['mean_sig_n'])}" if row["sd_sig_n"] == 0 else f"{row['mean_sig_n']:.2f}"
                ax.text(i, row["mean_sig_n"] + row["sd_sig_n"] + 0.35, txt, ha="center", va="bottom", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["label"])
            ax.set_ylabel("Significant sessions")
            ax.set_title("Figure 1. Correlation counts are indistinguishable from null")
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylim(0, max(plot_df["mean_sig_n"] + plot_df["sd_sig_n"]) + 4)
            save_figure(fig, f"{PRIMARY_FRAME}_binder_Figure1_correlation_counts")
            plot_df.to_csv(REPRO_CSV_DIR / f"{PRIMARY_FRAME}_binder_Figure1_counts.csv", index=False)
        """
    ),
    md_cell(
        """
        ## Figure 2

        Figure 2 can be reconstructed for any frame that has top/rest topology summaries.
        """
    ),
    code_cell(
        """
        frame_df = load_frame_session_summary(PRIMARY_FRAME).copy()
        plot_rows = []
        for _, row in frame_df.iterrows():
            plot_rows.append({"label": row["label"], "is_sig_session": row["is_sig_session"], "zone": "top", "lambda2": row["top_lambda2"], "aspl": row["top_aspl"]})
            plot_rows.append({"label": row["label"], "is_sig_session": row["is_sig_session"], "zone": "rest", "lambda2": row["rest_lambda2"], "aspl": row["rest_aspl"]})
        plot_df = pd.DataFrame(plot_rows)
        plot_df["lambda2_z"] = zscore(plot_df["lambda2"])
        plot_df["aspl_z"] = zscore(plot_df["aspl"])

        style = {
            (1, "top"): {"label": "Significant / top", "color": "#2563eb", "marker": "o", "alpha": 0.95},
            (1, "rest"): {"label": "Significant / rest", "color": "#93c5fd", "marker": "o", "alpha": 0.60},
            (0, "top"): {"label": "Non-significant / top", "color": "#dc2626", "marker": "^", "alpha": 0.95},
            (0, "rest"): {"label": "Non-significant / rest", "color": "#fca5a5", "marker": "^", "alpha": 0.60},
        }

        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        for _, row in frame_df.iterrows():
            rest = plot_df[(plot_df["label"] == row["label"]) & (plot_df["zone"] == "rest")].iloc[0]
            top = plot_df[(plot_df["label"] == row["label"]) & (plot_df["zone"] == "top")].iloc[0]
            color = "#2563eb" if row["is_sig_session"] == 1 else "#dc2626"
            ax.annotate("", xy=(top["aspl_z"], top["lambda2_z"]), xytext=(rest["aspl_z"], rest["lambda2_z"]),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.28, linewidth=1.0), zorder=1)
        for key, sty in style.items():
            sub = plot_df[(plot_df["is_sig_session"] == key[0]) & (plot_df["zone"] == key[1])]
            ax.scatter(sub["aspl_z"], sub["lambda2_z"], s=62, color=sty["color"], marker=sty["marker"],
                       alpha=sty["alpha"], edgecolor="white", linewidth=0.5, label=sty["label"])

        sig_top = plot_df[(plot_df["is_sig_session"] == 1) & (plot_df["zone"] == "top")]
        non_top = plot_df[(plot_df["is_sig_session"] == 0) & (plot_df["zone"] == "top")]
        add_cov_ellipse(ax, sig_top["aspl_z"], sig_top["lambda2_z"], "#2563eb", n_std=1.55, alpha=0.12)
        add_cov_ellipse(ax, non_top["aspl_z"], non_top["lambda2_z"], "#dc2626", n_std=1.55, alpha=0.12)
        ax.set_xlabel("ASPL (z)")
        ax.set_ylabel("lambda2 (z)")
        ax.set_title(f"Figure 2. Topology separates high-contribution structure\\n[{FRAME_REGISTRY[PRIMARY_FRAME]['label']}]")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9, loc="best")
        save_figure(fig, f"{PRIMARY_FRAME}_binder_Figure2_topology_separation")
        plot_df.to_csv(REPRO_CSV_DIR / f"{PRIMARY_FRAME}_binder_Figure2_points.csv", index=False)
        """
    ),
    md_cell(
        """
        ## Figure 3

        The bundled sliding-window aggregate is only available for the published primary frame.
        """
    ),
    code_cell(
        """
        reg = FRAME_REGISTRY[PRIMARY_FRAME]
        if reg["sliding_agg"] is None:
            print("Figure 3 is only available for PRIMARY_FRAME = '4ch_60corr_26task'.")
        else:
            sliding_agg_df = pd.read_csv(reg["sliding_agg"])
            t = sliding_agg_df["rel_t_sec"].to_numpy(float)
            mean_h1 = moving_average_centered(sliding_agg_df["mean_z_h1"], 3)
            mean_lambda2 = moving_average_centered(sliding_agg_df["mean_z_lambda2"], 3)
            mean_aspl = moving_average_centered(sliding_agg_df["mean_z_aspl"], 3)

            fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
            ax.axvspan(-2.0, -1.25, color="#ede9fe", alpha=0.55, zorder=0)
            ax.axvspan(-1.25, -0.25, color="#fee2e2", alpha=0.45, zorder=0)
            ax.axvspan(-0.25, 0.75, color="#dbeafe", alpha=0.45, zorder=0)
            ax.plot(t, mean_h1, color="#7c3aed", linewidth=2.4, label="H1-related persistence")
            ax.plot(t, mean_aspl, color="#dc2626", linewidth=2.4, label="ASPL")
            ax.plot(t, mean_lambda2, color="#2563eb", linewidth=2.4, label="lambda2")
            ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.set_xlabel("Time from peak-coupling task center (s)")
            ax.set_ylabel("Smoothed mean z-score")
            ax.set_title("Figure 3. Ordered temporal phase around peak coupling")
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, loc="best")
            save_figure(fig, f"{PRIMARY_FRAME}_binder_Figure3_temporal_phase")
            sliding_agg_df.to_csv(REPRO_CSV_DIR / f"{PRIMARY_FRAME}_binder_Figure3_sliding.csv", index=False)
        """
    ),
    md_cell(
        """
        ## Figure 4

        Figure 4 is also tied to the published primary frame because the bundled null-distribution summaries
        were precomputed for that frame.
        """
    ),
    code_cell(
        """
        reg = FRAME_REGISTRY[PRIMARY_FRAME]
        if reg["null_observed"] is None or reg["null_distribution"] is None:
            print("Figure 4 is only available for PRIMARY_FRAME = '4ch_60corr_26task'.")
        else:
            obs_df = pd.read_csv(reg["null_observed"])
            null_df = pd.read_csv(reg["null_distribution"])

            metrics = [
                ("delta_h1_zero", "delta H1=0"),
                ("delta_lambda2", "delta lambda2"),
                ("delta_aspl", "delta ASPL"),
            ]
            cond_style = {
                "false_session_matching": {"label": "False session matching", "color": "#d97706"},
                "random_fake_quantum": {"label": "Random fake quantum", "color": "#2563eb"},
            }

            fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), constrained_layout=True)
            for ax, (metric, title) in zip(axes, metrics):
                observed = float(obs_df[obs_df["metric"] == metric]["observed_diff_17v9"].iloc[0])
                vals_all = []
                for condition, sty in cond_style.items():
                    vals = pd.to_numeric(null_df.loc[null_df["condition"] == condition, f"median_diff_{metric}"], errors="coerce").dropna().to_numpy(float)
                    vals_all.append(vals)
                    ax.hist(vals, bins=24, density=True, alpha=0.40, color=sty["color"], label=sty["label"])
                    row = obs_df[(obs_df["condition"] == condition) & (obs_df["metric"] == metric)].iloc[0]
                    ax.text(
                        0.02, 0.92 if condition == "false_session_matching" else 0.77,
                        f"{sty['label']}\\n"
                        f"two-sided p = {float(row['p_two_sided_vs_observed']):.3g}\\n"
                        f"one-sided p = {float(row['p_one_sided_vs_observed']):.3g}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
                        bbox=dict(boxstyle="round", alpha=0.10),
                    )
                ax.axvline(observed, color="black", linewidth=3.0)
                ax.axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.45)
                lo = min(np.min(v) for v in vals_all if len(v) > 0)
                hi = max(np.max(v) for v in vals_all if len(v) > 0)
                pad = max(0.04, 0.08 * (hi - lo if hi > lo else 1.0))
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_title(title)
                ax.set_xlabel("Observed median(sig) - median(non-sig)")
                ax.grid(axis="y", alpha=0.22)
            axes[0].set_ylabel("Null density")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
            fig.suptitle("Figure 4. Structural signature is not reproduced by null controls", fontsize=13)
            save_figure(fig, f"{PRIMARY_FRAME}_binder_Figure4_null_robustness")
        """
    ),
    md_cell(
        """
        ## Figure 5

        Figure 5 compares any set of precomputed frames listed in `FRAME_COMPARISON`.
        """
    ),
    code_cell(
        """
        rows = []
        for frame_key in FRAME_COMPARISON:
            df = load_frame_session_summary(frame_key)
            sig = df[df["is_sig_session"] == 1]
            non = df[df["is_sig_session"] == 0]
            row = {"condition": FRAME_REGISTRY[frame_key]["label"]}
            for metric in ["delta_h1_zero", "delta_lambda2", "delta_aspl"]:
                med, lo, hi = bootstrap_median_diff(sig[metric].to_numpy(float), non[metric].to_numpy(float))
                row[metric] = med
                row[f"{metric}_lo"] = lo
                row[f"{metric}_hi"] = hi
            rows.append(row)
        frame_comparison_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        x = np.arange(len(frame_comparison_df))
        spec = [
            ("delta_h1_zero", "#7c3aed", "delta H1=0"),
            ("delta_lambda2", "#2563eb", "delta lambda2"),
            ("delta_aspl", "#dc2626", "delta ASPL"),
        ]
        for metric, color, label in spec:
            y = frame_comparison_df[metric].to_numpy(float)
            yerr = np.vstack([
                y - frame_comparison_df[f"{metric}_lo"].to_numpy(float),
                frame_comparison_df[f"{metric}_hi"].to_numpy(float) - y,
            ])
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4, color=color, label=label)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(frame_comparison_df["condition"], rotation=10)
        ax.set_ylabel("Median(sig) - median(non-sig)")
        ax.set_title("Figure 5. Sensitivity to observation frame")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, loc="best")
        save_figure(fig, "binder_Figure5_frame_dependence")
        frame_comparison_df.to_csv(REPRO_CSV_DIR / "binder_Figure5_frame_dependence_values.csv", index=False)
        """
    ),
    md_cell(
        """
        ## Output summary

        All Binder-demo outputs are written to:

        - `Repro_CSV`
        - `Repro_Figure`
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUT_PATH)
