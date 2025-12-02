"""
analyze_tradeoff.py

Phase 2: Flow vs. Stiffness trade-off study.

- Loads results/flow_metrics/flow_metrics.csv
- Filters to synthetic samples
- Uses Porosity to define:
    density = 1 - porosity
    stiffness_potential = (density ** 2)
- Normalizes both axes:
    x = Relative Flow Rate (Q / Q_max)
    y = Normalized Stiffness Potential (E / E_solid)
- Saves scatter plot to results/flow_stiffness_tradeoff.png
- Prints top candidate designs by (flow * stiffness) score
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "flow_metrics" / "flow_metrics.csv"
OUT_PLOT = ROOT / "results" / "flow_stiffness_tradeoff.png"
OUT_CSV = ROOT / "results" / "flow_stiffness_candidates.csv"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {METRICS_PATH}. "
            "Run flow_simulation.py first."
        )

    df = pd.read_csv(METRICS_PATH)

    # Expect columns: Mean_K, Mean_dP/dy, FlowRate, Porosity, Anisotropy, Type
    required_cols = {"FlowRate", "Porosity", "Type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in flow_metrics.csv: {missing}")

    # Filter to synthetic structures only
    mask_syn = df["Type"].astype(str).str.lower() == "synthetic"
    df_syn = df[mask_syn].copy()
    if df_syn.empty:
        raise RuntimeError("No synthetic samples found (Type == 'synthetic').")

    # Porosity is assumed to be a fraction in [0, 1].
    # Quick sanity check; if you see ~10–50, it is likely in percent.
    por_min = df_syn["Porosity"].min()
    por_max = df_syn["Porosity"].max()
    print(f"Porosity range (synthetic): [{por_min:.4f}, {por_max:.4f}]")

    # If you *know* it's in percent, you could uncomment this:
    # if por_max > 1.0:
    #     df_syn["Porosity"] = df_syn["Porosity"] / 100.0

    porosity = df_syn["Porosity"].values
    density = 1.0 - porosity

    # Stiffness potential heuristic: E ∝ ρ^2
    stiffness_raw = density ** 2

    # Normalize to make axes unitless and comparable
    stiffness_norm = stiffness_raw / stiffness_raw.max()

    flow_raw = df_syn["FlowRate"].values
    flow_norm = flow_raw / flow_raw.max()

    df_syn["density"] = density
    df_syn["stiffness_potential"] = stiffness_raw
    df_syn["stiffness_norm"] = stiffness_norm
    df_syn["flow_norm"] = flow_norm

    # Simple combined score to find "interesting" candidates:
    # high flow *and* high stiffness.
    df_syn["score_flow_x_stiffness"] = df_syn["flow_norm"] * df_syn["stiffness_norm"]
    df_syn_sorted = df_syn.sort_values(
        "score_flow_x_stiffness", ascending=False
    ).reset_index(drop=True)

    # Save top candidates to CSV for inspection
    top_k = 10
    df_syn_sorted.head(top_k).to_csv(OUT_CSV, index=False)
    print(f"✅ Saved top {top_k} candidate designs → {OUT_CSV}")

    # Scatter plot: Flow vs. Stiffness
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df_syn["flow_norm"],
        df_syn["stiffness_norm"],
        alpha=0.4,
        s=15,
        label="Synthetic samples",
    )

    # Highlight top-k on the plot
    top = df_syn_sorted.head(top_k)
    plt.scatter(
        top["flow_norm"],
        top["stiffness_norm"],
        s=40,
        edgecolors="black",
        linewidths=0.7,
        label=f"Top {top_k} (flow × stiffness)",
    )

    plt.xlabel("Relative Flow Rate  (Q / Q_max)")
    plt.ylabel("Normalized Stiffness Potential  (E / E_solid)")
    plt.title("Flow–Stiffness Trade-off for Synthetic Microstructures")
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUT_PLOT.parent, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=200)
    print(f"✅ Trade-off plot saved → {OUT_PLOT}")

    # Print a small text summary of the top candidates
    print("\nTop candidate designs (by flow × stiffness score):")
    cols_to_show = [
        "FlowRate",
        "Porosity",
        "density",
        "stiffness_potential",
        "flow_norm",
        "stiffness_norm",
        "score_flow_x_stiffness",
    ]
    print(df_syn_sorted[cols_to_show].head(top_k).to_string(index=False))


if __name__ == "__main__":
    main()
