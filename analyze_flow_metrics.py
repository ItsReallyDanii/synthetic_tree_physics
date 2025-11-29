import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, ks_2samp

# -----------------------------
# 🧠 CONFIG
# -----------------------------
CSV_PATH = "results/flow_metrics/flow_metrics.csv"
OUTPUT_DIR = "results/metric_distributions"
REPORT_PATH = "results/physics_validation_report.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 📥 LOAD DATA
# -----------------------------
print(f"📂 Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Split real vs synthetic
real = df[df["Type"].str.lower() == "real"]
synthetic = df[df["Type"].str.lower() == "synthetic"]

# Columns to analyze
metrics = ["Mean_K", "Mean_dP/dy", "FlowRate", "Porosity", "Anisotropy"]

# -----------------------------
# 📊 ANALYSIS
# -----------------------------
report = {}

for metric in metrics:
    if metric not in df.columns:
        print(f"⚠️ Skipping {metric} — not found in CSV.")
        continue

    # Replace slash in filenames
    safe_metric = metric.replace("/", "_")

    # Plot distributions
    plt.figure(figsize=(6, 4))
    plt.hist(real[metric], bins=10, alpha=0.6, label="Real", density=True)
    plt.hist(synthetic[metric], bins=10, alpha=0.6, label="Synthetic", density=True)
    plt.title(f"{metric}: Real vs Synthetic")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"{safe_metric}_distribution.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved: {save_path}")

    # Statistical tests
    t_stat, t_p = ttest_ind(real[metric], synthetic[metric], equal_var=False, nan_policy="omit")
    ks_stat, ks_p = ks_2samp(real[metric], synthetic[metric])

    report[metric] = {
        "real_mean": np.nanmean(real[metric]),
        "synthetic_mean": np.nanmean(synthetic[metric]),
        "t_p_value": round(t_p, 5),
        "ks_p_value": round(ks_p, 5),
    }

# -----------------------------
# 💾 SAVE REPORT
# -----------------------------
report_df = pd.DataFrame(report).T
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
report_df.to_csv(REPORT_PATH)
print(f"\n🧾 Physics validation report saved → {REPORT_PATH}\n")

# -----------------------------
# 🧮 AUTO SUMMARY
# -----------------------------
similar_t = (report_df["t_p_value"] > 0.05).sum()
similar_ks = (report_df["ks_p_value"] > 0.05).sum()
total = len(report_df)

summary = f"""
🌿 Flow Physics Validation Summary
-----------------------------------
✅ Metrics analyzed: {total}
✅ Metrics statistically similar (T-test): {similar_t}/{total}
✅ Metrics distribution-similar (KS-test): {similar_ks}/{total}

Interpretation:
If most p-values > 0.05, your generated xylem behaves
statistically like real xylem under flow physics.
"""

print(summary)
print("\n📊 Detailed Metric Comparison:\n")
print(report_df)