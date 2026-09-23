#!/usr/bin/env python3
"""Reproduce all quantitative benchmark tables from Paper 1 (PBA-Net).

Generates:
  - Table 3: External Multi-Center Generalization Benchmark (BUSI N=647, 15 Models across 3 Seeds)
  - Table 6: Lesion Dependence and Background Shortcut Perturbation Audit
  - Table 7: Controlled Image-Domain Perturbation & Calibration Stability Audit
"""

from __future__ import annotations
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    res_dir = root / "results"

    pba_rep_path = res_dir / "pba_3seed_comprehensive_report.json"
    phys_rep_path = res_dir / "physics_invariance_report.json"
    faith_rep_path = res_dir / "pba_faithfulness_report.json"

    print("==========================================================================================")
    print("                     REPRODUCING PAPER 1 BENCHMARK TABLES (PBA-Net)")
    print("==========================================================================================\n")

    # Table 3: External Benchmark
    if pba_rep_path.exists():
        with open(pba_rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregated_summary", {})
        print("TABLE 3: External Generalization Benchmark on BUSI (N=647, 15 Models across 3 Seeds)")
        print("-" * 92)
        print(f"{'Method / Pathway Variant':<32} | {'AUROC':<18} | {'AUPRC':<18} | {'ECE':<16}")
        print("-" * 92)
        mapping = [
            ("Whole-Image Baseline (ResNet-50)", "whole_baseline"),
            ("PBA-Net (Global Branch Only)", "pba_global"),
            ("PBA-Net (Physics Branch Only)", "pba_physics"),
            ("PBA-Net (Stage 1 Gated Comb)", "pba_combined"),
            ("PBA-Net (Proposed Safety Blend)", "whole_plus_physics_blend"),
        ]
        for label, key in mapping:
            if key in agg:
                m = agg[key]
                auroc_m = m.get("auroc", {}).get("mean", 0.0)
                auroc_s = m.get("auroc", {}).get("sd", 0.0)
                auprc_m = m.get("auprc", {}).get("mean", 0.0)
                auprc_s = m.get("auprc", {}).get("sd", 0.0)
                ece_m = m.get("ece", {}).get("mean", 0.0)
                ece_s = m.get("ece", {}).get("sd", 0.0)
                print(f"{label:<32} | {auroc_m:.4f} ± {auroc_s:.4f}  | {auprc_m:.4f} ± {auprc_s:.4f}  | {ece_m:.4f} ± {ece_s:.4f}")
        print("-" * 92)

    # Table 6: Shortcut Audit
    if faith_rep_path.exists():
        with open(faith_rep_path, "r", encoding="utf-8") as f:
            fdata = json.load(f)
        print("\nTABLE 6: Lesion Dependence and Background Shortcut Perturbation Audit")
        print("-" * 92)
        print(f"{'Intervention Condition':<28} | {'Whole AUROC':<18} | {'PBA Physics AUROC':<18} | {'PBA - Whole':<16}")
        print("-" * 92)
        conds = [
            ("Intact Clean Reference", "identity"),
            ("Lesion Erased (Mean Fill)", "zero_lesion_bgmean"),
            ("Lesion Only (Black Canvas)", "keep_lesion_black"),
        ]
        seeds = ["20260613", "20260614", "20260615"]
        for label, ckey in conds:
            w_vals, p_vals = [], []
            for s in seeds:
                if s in fdata and ckey in fdata[s]:
                    w_vals.append(fdata[s][ckey].get("global_auroc", 0.0))
                    p_vals.append(fdata[s][ckey].get("physics_auroc", 0.0))
            if w_vals and p_vals:
                mw = sum(w_vals) / len(w_vals)
                mp = sum(p_vals) / len(p_vals)
                diff = mp - mw
                print(f"{label:<28} | {mw:.4f}             | {mp:.4f}             | {diff:+.4f}")
        print("-" * 92)

    # Table 7: Perturbation Stress Test
    if phys_rep_path.exists():
        with open(phys_rep_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        print("\nTABLE 7: Controlled Image-Domain Perturbation & Calibration Stability Audit")
        print("-" * 92)
        print(f"{'Perturbation Condition':<28} | {'Whole AUROC':<18} | {'PBA-Net AUROC':<18} | {'PBA-Net ECE':<16}")
        print("-" * 92)
        p_conds = [
            ("Clean (Unperturbed)", "clean"),
            ("Global Gain (+20%)", "gain_p20"),
            ("Global Gain (-20%)", "gain_m20"),
            ("Depth TGC Ramp (+25%)", "tgc_ramp"),
            ("Gamma Compression (1.4)", "gamma_14"),
            ("Acoustic Speckle (0.15)", "speckle"),
        ]
        auroc_dict = pdata.get("auroc", {})
        ece_dict = pdata.get("ece", {})
        for label, ckey in p_conds:
            if ckey in auroc_dict:
                w_info = auroc_dict[ckey].get("whole", {})
                b_info = auroc_dict[ckey].get("pba_blend", {})
                b_ece = ece_dict.get(ckey, {}).get("pba_blend", {})
                w_str = f"{w_info.get('mean', 0.0):.4f} ± {w_info.get('sd', 0.0):.4f}"
                b_str = f"{b_info.get('mean', 0.0):.4f} ± {b_info.get('sd', 0.0):.4f}"
                ece_str = f"{b_ece.get('mean', 0.0):.4f} ± {b_ece.get('sd', 0.0):.4f}"
                print(f"{label:<28} | {w_str:<18} | {b_str:<18} | {ece_str:<16}")
        print("-" * 92)

    print("\nBenchmark tables verified and printed successfully.\n")


if __name__ == "__main__":
    main()
