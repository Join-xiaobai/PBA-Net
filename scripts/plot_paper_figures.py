#!/usr/bin/env python3
"""Generate publication-grade figures for PBA-Net.

Plots:
  - Figure 4: Controlled Ultrasound Image-Domain Perturbation & Stability Audit
  - Figure 5: Lesion-Dependence and Peripheral Shortcut Perturbation Audit
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

root = Path(__file__).resolve().parent.parent
res_dir = root / "results"
out_dir = root / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10.5
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def plot_perturbation_audit():
    phys_rep_path = res_dir / "physics_invariance_report.json"
    if not phys_rep_path.exists():
        print(f"Skipping perturbation figure: missing {phys_rep_path}")
        return

    with open(phys_rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conditions = ["Clean", "Gain +20%", "Gain -20%", "TGC Ramp", "Gamma 1.4", "Speckle"]
    keys = ["clean", "gain_p20", "gain_m20", "tgc_ramp", "gamma_14", "speckle"]

    whole_auroc = [data["auroc"][k]["whole"]["mean"] for k in keys]
    whole_std = [data["auroc"][k]["whole"]["sd"] for k in keys]
    pba_auroc = [data["auroc"][k]["pba_blend"]["mean"] for k in keys]
    pba_std = [data["auroc"][k]["pba_blend"]["sd"] for k in keys]
    pba_ece = [data["ece"][k]["pba_blend"]["mean"] for k in keys]

    x = np.arange(len(conditions))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # Panel 1: AUROC
    ax1.bar(x - width/2, whole_auroc, width, yerr=whole_std, label="Whole-Image Baseline", color="#4C72B0", capsize=3, alpha=0.85)
    ax1.bar(x + width/2, pba_auroc, width, yerr=pba_std, label="PBA-Net (Ours)", color="#2B5C8F", capsize=3, alpha=0.95)
    ax1.set_ylabel("External AUROC (BUSI N=647)")
    ax1.set_title("(a) Discrimination Retention under Acquisition Variations")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=15)
    ax1.set_ylim([0.70, 0.93])
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="lower left", framealpha=0.9)

    # Panel 2: ECE
    whole_ece = [data["ece"][k]["whole"]["mean"] for k in keys]
    ax2.plot(x, whole_ece, "o--", color="#4C72B0", label="Whole-Image Baseline", linewidth=2, markersize=6)
    ax2.plot(x, pba_ece, "s-", color="#2B5C8F", label="PBA-Net (Ours)", linewidth=2.2, markersize=6)
    ax2.set_ylabel("Expected Calibration Error (ECE)")
    ax2.set_title("(b) Probability Calibration Stability under Image Variations")
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=15)
    ax2.set_ylim([0.02, 0.14])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    out_path = out_dir / "Figure4_Perturbation_Audit.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_shortcut_audit():
    faith_rep_path = res_dir / "pba_faithfulness_report.json"
    if not faith_rep_path.exists():
        print(f"Skipping shortcut figure: missing {faith_rep_path}")
        return

    with open(faith_rep_path, "r", encoding="utf-8") as f:
        fdata = json.load(f)

    conds = [("Intact", "identity"), ("Lesion Erased", "zero_lesion_bgmean"), ("Lesion Only", "keep_lesion_black")]
    seeds = ["20260613", "20260614", "20260615"]

    labels = ["Intact Clean", "Lesion Erased\n(Mean Fill)", "Lesion Only\n(Black Background)"]
    whole_means, phys_means = [], []
    for _, ckey in conds:
        w_vals = [fdata[s][ckey]["global_auroc"] for s in seeds if s in fdata and ckey in fdata[s]]
        p_vals = [fdata[s][ckey]["physics_auroc"] for s in seeds if s in fdata and ckey in fdata[s]]
        whole_means.append(np.mean(w_vals))
        phys_means.append(np.mean(p_vals))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    ax.bar(x - width/2, whole_means, width, label="Whole-Image Baseline", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, phys_means, width, label="PBA-Net Physics Branch", color="#8172B3", alpha=0.95)
    ax.set_ylabel("External AUROC (BUSI N=647)")
    ax.set_title("Lesion-Dependence & Background Shortcut Audit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.75, 0.92])
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out_path = out_dir / "Figure5_Shortcut_Audit.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    print("Generating publication figures...")
    plot_perturbation_audit()
    plot_shortcut_audit()
    print("Done!")


if __name__ == "__main__":
    main()
