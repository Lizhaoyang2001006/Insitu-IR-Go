#!/usr/bin/env python3
"""
Step 2 & Step 3 (Reviewer-Hardened, Sample-Agnostic, Full Deconvolution Curve Disclosure):
  - Step 2: Correlation-gated Gas-Phase H2O(g) nu2 Orthogonal Projection Subtraction
      * Template source auto-selection maximizing high-pass std on [1700, 1800] cm^-1
      * 55-pt cubic Savitzky-Golay high-pass residual, zeroed <1350 cm^-1, linear taper on [1350, 1365] cm^-1
      * Unit L2 normalization on anchor window Omega_anc = [1700.0, 1800.0] cm^-1
      * Signed projection beta_raw = <S_hp, W>_Omega_anc gated by |r| >= corr_gate (default 0.50)
      * Post-subtraction median baseline re-zeroing on Omega_bl = [1750.0, 1800.0] cm^-1
      * High-frequency ripple evaluation via 41-pt cubic SG residual std on [1480.0, 1720.0] cm^-1
  - Step 3: Three-metric cross-validated Tier A quantification & full curve archival:
      * (1) Shared-basis 6-Gaussian box-constrained least-squares deconvolution on [1475.0, 1750.0] cm^-1:
            I_k = sqrt(2*pi) * c_k * sigma_k, R_decon = I(1596)/I(1566), plus R^2 and RMSE;
            exports wavenumber-resolved deconvolution curve CSVs for EVERY processed spectrum
            (under decon_curves/<Dataset>/), aligned endpoint SI CSVs, manifest index, and grid QC plots.
      * (2) Peak-top sensitivity ratio R_1590 = A(1590±2)/A(1568±2) (zero-parameter positive control)
      * (3) Classical literature direct ratio R_1600 = A(1600±2)/A(1568±2) + secondary ratios

Usage:
  python3 scripts/step2_step3_wvclean_and_ratios.py --in-dir ./output/step1A --out-dir ./output/step2_3
"""
import argparse
import glob
import math
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

# Universal shared physical Gaussian basis on 1475-1750 cm^-1: (center_cm-1, sigma_cm-1, plot_label, csv_col_name)
SHARED_BASIS_PEAKS = [
    (1638.0, 24.0, "H₂O / MEAH⁺ (1638)",  "Peak1_1638_H2O_MEAH"),
    (1596.0, 15.0, "*MEA (1596)",         "Peak2_1596_MEA"),
    (1566.0, 15.0, "MEACOO⁻ ν_as (1566)", "Peak3_1566_MEACOO_as"),
    (1528.0, 18.0, "Shoulder (1528)",     "Peak4_1528_Shoulder"),
    (1487.0, 15.0, "MEACOO⁻ ν_s (1487)",  "Peak5_1487_MEACOO_s"),
    (1438.0, 18.0, "1432 Wing (1438)",    "Peak6_1438_Wing"),
]
DECON_WIN = (1475.0, 1750.0)
OFFSET_BOUND = 0.015
PALETTE = ["#333333", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def parse_num(s):
    m = re.search(r"([0-9]+\.?[0-9]*)", str(s))
    return float(m.group(1)) if m else np.nan


def band_mean(wn, y, center, hw=2.0):
    m = (wn >= center - hw) & (wn <= center + hw)
    return float(np.mean(y[m]))


def build_shared_basis(xf):
    B = np.zeros((len(xf), len(SHARED_BASIS_PEAKS) + 1), dtype=float)
    for k, (mu, sig, _, _) in enumerate(SHARED_BASIS_PEAKS):
        B[:, k] = np.exp(-0.5 * ((xf - mu) / sig) ** 2)
    B[:, -1] = 1.0  # bounded flat baseline offset c_0 in [-0.015, +0.015]
    return B


def deconvolve_spectrum(xf, B, yf):
    n_p = len(SHARED_BASIS_PEAKS)
    res = least_squares(
        lambda c: B @ c - yf,
        [0.02] * n_p + [0.0],
        bounds=([0.0] * n_p + [-OFFSET_BOUND], [np.inf] * n_p + [OFFSET_BOUND]),
    )
    c = res.x
    y_hat = B @ c
    resid = yf - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    areas = [c[k] * SHARED_BASIS_PEAKS[k][1] * np.sqrt(2.0 * np.pi) for k in range(n_p)]
    ratio_decon = areas[1] / areas[2] if areas[2] > 1e-8 else np.nan
    return c, y_hat, resid, r2, rmse, areas, ratio_decon


def build_decon_curve_df(x_s, y_s, y_hat, resid, c):
    """Construct a 13-column Origin-ready DataFrame for a single deconvolved spectrum."""
    c0 = float(c[-1])
    d = {
        "Wavenumber (cm-1)": np.round(x_s, 5),
        "Experimental_S_clean": np.round(y_s, 7),
        "Sum_Fit_S_hat": np.round(y_hat, 7),
        "Residual": np.round(resid, 7),
        "Baseline_Offset_c0": np.round(np.full_like(x_s, c0), 7),
        "Experimental_Zeroed": np.round(y_s - c0, 7),
        "Sum_Peaks_Zeroed": np.round(y_hat - c0, 7),
    }
    for p_i, (mu, sig, _, col_name) in enumerate(SHARED_BASIS_PEAKS):
        comp = c[p_i] * np.exp(-0.5 * ((x_s - mu) / sig) ** 2)
        d[col_name] = np.round(comp, 7)
    return pd.DataFrame(d)


def format_axis_tag(key, cname, raw_val, first_val):
    if key.endswith("_Hold") and np.isfinite(raw_val) and np.isfinite(first_val):
        t_rel = raw_val - first_val
        return f"t{t_rel:05.2f}min", round(float(t_rel), 3), "min"
    if key.endswith("_Ramp") and np.isfinite(raw_val):
        if abs(raw_val - round(raw_val)) < 1e-6:
            return f"T{int(round(raw_val)):03d}C", round(float(raw_val), 3), "C"
        return f"T{raw_val:05.1f}C", round(float(raw_val), 3), "C"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cname)).strip("_") or "spec"
    return safe, raw_val, ""


def load_step1a_files(in_dir):
    paths = sorted(glob.glob(os.path.join(in_dir, "*_1000-1800_bl1A.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_1000-1800_bl1A.csv files found in {in_dir}")
    dfs = {}
    for p in paths:
        key = os.path.basename(p).replace("_1000-1800_bl1A.csv", "")
        dfs[key] = pd.read_csv(p, encoding="utf-8-sig")
    return dfs


def discover_mode_keys(dfs, mode_suffix):
    tag = f"_{mode_suffix}"
    matched = [k for k in dfs if k.endswith(tag)]
    blanks = sorted([k for k in matched if k[:-len(tag)].lower() == "blank"])
    others = sorted([k for k in matched if k[:-len(tag)].lower() != "blank"])
    return blanks + others


def pick_vapor_template_source(dfs, requested_key=None):
    if requested_key and requested_key in dfs:
        return requested_key
    best_key, best_hf = None, -1.0
    for k, df in dfs.items():
        wn = df.iloc[:, 0].values.astype(float)
        m_anc = (wn >= 1700.0) & (wn <= 1800.0)
        tail = df.iloc[:, max(1, df.shape[1] - 8):].values.astype(float)
        avg = np.mean(tail, axis=1)
        hf = float(np.std((avg - savgol_filter(avg, 55, 3))[m_anc]))
        if hf > best_hf:
            best_hf = hf
            best_key = k
    return best_key


def main():
    parser = argparse.ArgumentParser(
        description="Step 2 (correlation-gated H2O vapor subtraction) & Step 3 (3-metric ratios + full deconvolution curve CSVs)."
    )
    parser.add_argument("--in-dir", default="./output/step1A", help="Directory containing Step 1A *_1000-1800_bl1A.csv files.")
    parser.add_argument("--out-dir", default="./output/step2_3", help="Output directory for Step 2 & Step 3 files and plots.")
    parser.add_argument("--corr-gate", type=float, default=0.50, help="Minimum |r| on 1700-1800 cm^-1 to apply H2O(g) subtraction.")
    parser.add_argument("--template-key", default=None, help="Optional dataset key to extract H2O(g) template W(nu) (default: auto).")
    parser.add_argument("--max-hold-min", type=float, default=10.6, help="Aligned isothermal hold cutoff in minutes.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    decon_root = os.path.join(args.out_dir, "decon_curves")
    os.makedirs(decon_root, exist_ok=True)
    dfs = load_step1a_files(args.in_dir)

    # ==========================================================================
    # STEP 2: Correlation-Gated Gas-Phase H2O(g) Orthogonal Projection
    # ==========================================================================
    print("=" * 88)
    print(f"STEP 2: Correlation-Gated H2O(g) Subtraction (|r| >= {args.corr_gate:.2f} on 1700-1800 cm^-1)")
    print("=" * 88)

    tmpl_key = pick_vapor_template_source(dfs, args.template_key)
    wn = dfs[tmpl_key].iloc[:, 0].values.astype(float)
    mat_tmpl = dfs[tmpl_key].iloc[:, 1:].values.astype(float)
    n_tail = min(8, mat_tmpl.shape[1])
    avg_wv = np.mean(mat_tmpl[:, -n_tail:], axis=1)
    hp_raw = avg_wv - savgol_filter(avg_wv, 55, 3)

    W_raw = np.zeros_like(wn)
    m_wv = (wn >= 1350.0) & (wn <= 1800.3)
    W_raw[m_wv] = hp_raw[m_wv]
    m_taper = (wn >= 1350.0) & (wn <= 1365.0)
    W_raw[m_taper] *= (wn[m_taper] - 1350.0) / 15.0

    m_anchor = (wn >= 1700.0) & (wn <= 1800.0)
    W = W_raw / float(np.linalg.norm(W_raw[m_anchor]))

    pd.DataFrame({
        "Wavenumber (cm-1)": np.round(wn, 5),
        "W_H2O_vapor_template": np.round(W, 7)
    }).to_csv(os.path.join(args.out_dir, "Step2_H2O_vapor_template_W.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Template W(nu) extracted from '{tmpl_key}' (last {n_tail} spectra, anchor_pts={int(m_anchor.sum())})")

    m_eval = (wn >= 1480.0) & (wn <= 1720.0)
    m_bl = (wn >= 1750.0) & (wn <= 1800.0)
    step2_audit = []
    clean_dfs = {}

    for key, df in dfs.items():
        cols = list(df.columns)
        w_this = df.iloc[:, 0].values.astype(float)
        out_dict = {cols[0]: np.round(w_this, 5)}
        for j in range(1, len(cols)):
            cname = cols[j]
            s_raw = df.iloc[:, j].values.astype(float)
            s_hp = s_raw - savgol_filter(s_raw, 55, 3)

            r_anc = float(np.corrcoef(s_hp[m_anchor], W[m_anchor])[0, 1])
            beta_raw = float(np.dot(s_hp[m_anchor], W[m_anchor]))
            gate_passed = bool(abs(r_anc) >= args.corr_gate)
            beta_app = beta_raw if gate_passed else 0.0

            s_cln = s_raw - beta_app * W
            bl_res = float(np.median(s_cln[m_bl]))
            s_cln = s_cln - bl_res

            rms0 = float(np.std((s_raw - savgol_filter(s_raw, 41, 3))[m_eval]))
            rms1 = float(np.std((s_cln - savgol_filter(s_cln, 41, 3))[m_eval]))
            red = (1.0 - rms1 / rms0) * 100.0 if rms0 > 0 else 0.0

            out_dict[cname] = np.round(s_cln, 7)
            step2_audit.append({
                "Dataset": key,
                "Spectrum": cname,
                "corr_1700_1800": round(r_anc, 4),
                "beta_raw": round(beta_raw, 6),
                "gate_passed": gate_passed,
                "beta_applied": round(beta_app, 6),
                "Post_BL_Rezero": round(bl_res, 6),
                "HF_RMS_Before": round(rms0, 6),
                "HF_RMS_After": round(rms1, 6),
                "HF_RMS_Reduction_Pct": round(red, 2),
            })

        cdf = pd.DataFrame(out_dict)
        clean_dfs[key] = cdf
        cdf.to_csv(os.path.join(args.out_dir, f"{key}_1000-1800_bl1A_wvclean.csv"), index=False, encoding="utf-8-sig")

    audit2_df = pd.DataFrame(step2_audit)
    audit2_df.to_csv(os.path.join(args.out_dir, "Step2_H2O_Vapor_Audit.csv"), index=False, encoding="utf-8-sig")
    for key in dfs:
        sub = audit2_df[audit2_df["Dataset"] == key]
        n_pass = int(sub["gate_passed"].sum())
        print(
            f"  {key:24s}: gated={n_pass:2d}/{len(sub):2d} | corr=[{sub['corr_1700_1800'].min():+.2f},{sub['corr_1700_1800'].max():+.2f}] | "
            f"beta_app=[{sub['beta_applied'].min():+.4f},{sub['beta_applied'].max():+.4f}] | "
            f"HF_RMS: {sub['HF_RMS_Before'].mean():.5f}->{sub['HF_RMS_After'].mean():.5f} (avg {sub['HF_RMS_Reduction_Pct'].mean():+.1f}%)"
        )

    # ==========================================================================
    # STEP 3: Three-Metric Quantification + Per-Spectrum Deconvolution Curve CSVs
    # ==========================================================================
    print("\n" + "=" * 88)
    print("STEP 3: Three-Metric Quantification & Full Wavenumber-Resolved Deconvolution Export")
    print("=" * 88)

    m_fit = (wn >= DECON_WIN[0]) & (wn <= DECON_WIN[1])
    xf = wn[m_fit]
    B = build_shared_basis(xf)
    decon_manifest_rows = []

    def quantify_dataset(df, key):
        w = df.iloc[:, 0].values.astype(float)
        m_f = (w >= DECON_WIN[0]) & (w <= DECON_WIN[1])
        ds_curve_dir = os.path.join(decon_root, key)
        os.makedirs(ds_curve_dir, exist_ok=True)
        first_num = parse_num(df.columns[1]) if df.shape[1] > 1 else np.nan

        rows = []
        fit_records = []
        for j in range(1, df.shape[1]):
            spec_idx = j - 1
            cname = df.columns[j]
            raw_num = parse_num(cname)
            y = df.iloc[:, j].values.astype(float)
            a1634 = band_mean(w, y, 1634.0)
            a1600 = band_mean(w, y, 1600.0)
            a1590 = band_mean(w, y, 1590.0)
            a1568 = band_mean(w, y, 1568.0)
            a1487 = band_mean(w, y, 1487.0)
            c, y_hat, resid, fit_r2, fit_rmse, areas, r_decon = deconvolve_spectrum(w[m_f], B, y[m_f])
            fit_records.append((cname, w[m_f], y[m_f], y_hat, resid, c, fit_r2, fit_rmse))

            # Export per-spectrum deconvolution curve CSV
            axis_tag, axis_val, axis_unit = format_axis_tag(key, cname, raw_num, first_num)
            curve_fn = f"{key}_spec{spec_idx:02d}_{axis_tag}_decon.csv"
            curve_rel = os.path.join("decon_curves", key, curve_fn)
            curve_df = build_decon_curve_df(w[m_f], y[m_f], y_hat, resid, c)
            curve_df.to_csv(os.path.join(ds_curve_dir, curve_fn), index=False, encoding="utf-8-sig")

            decon_manifest_rows.append({
                "Dataset": key,
                "SpecIdx": spec_idx,
                "Label": cname,
                "Axis_Value": axis_val,
                "Axis_Unit": axis_unit,
                "Amp_c1_1638": round(float(c[0]), 7),
                "Amp_c2_1596": round(float(c[1]), 7),
                "Amp_c3_1566": round(float(c[2]), 7),
                "Amp_c4_1528": round(float(c[3]), 7),
                "Amp_c5_1487": round(float(c[4]), 7),
                "Amp_c6_1438": round(float(c[5]), 7),
                "Offset_c0": round(float(c[6]), 7),
                "Area_1638_H2O_MEAH": round(float(areas[0]), 6),
                "Area_1596_MEA": round(float(areas[1]), 6),
                "Area_1566_MEACOO": round(float(areas[2]), 6),
                "Area_1528_Shoulder": round(float(areas[3]), 6),
                "Area_1487_MEACOO": round(float(areas[4]), 6),
                "Area_1438_Wing": round(float(areas[5]), 6),
                "R_decon_1596_1566": round(float(r_decon), 6),
                "Decon_Fit_R2": round(float(fit_r2), 6),
                "Decon_Fit_RMSE": round(float(fit_rmse), 6),
                "Curve_CSV": curve_rel.replace("\\", "/"),
            })

            rows.append({
                "Dataset": key,
                "Label": cname,
                "NumVal": raw_num,
                "A_1634": a1634,
                "A_1600": a1600,
                "A_1590": a1590,
                "A_1568": a1568,
                "A_1487": a1487,
                "R_1600_1568": a1600 / a1568,
                "R_1590_1568": a1590 / a1568,
                "R_decon_1596_1566": r_decon,
                "Decon_Fit_R2": fit_r2,
                "Decon_Fit_RMSE": fit_rmse,
                "Decon_Offset_c0": float(c[-1]),
                "Area_1638_H2O_MEAH": areas[0],
                "Area_1596_MEA": areas[1],
                "Area_1566_MEACOO": areas[2],
                "Area_1528_Shoulder": areas[3],
                "Area_1487_MEACOO": areas[4],
                "Area_1438_Wing": areas[5],
                "R_1487_1568": a1487 / a1568,
                "R_1634_1568": a1634 / a1568,
            })
        res = pd.DataFrame(rows)
        for col in ("R_1600_1568", "R_1590_1568", "R_decon_1596_1566"):
            res[f"d{col}"] = res[col] - res[col].iloc[0]
        return res, fit_records

    ramp_keys = discover_mode_keys(clean_dfs, "Ramp")
    hold_keys = discover_mode_keys(clean_dfs, "Hold")
    ordered_all_keys = hold_keys + ramp_keys + [k for k in clean_dfs if k not in hold_keys and k not in ramp_keys]

    all_quant = {}
    all_fit_records = {}
    for k in ordered_all_keys:
        qdf, frec = quantify_dataset(clean_dfs[k], k)
        all_quant[k] = qdf
        all_fit_records[k] = frec

    pd.DataFrame(decon_manifest_rows).to_csv(
        os.path.join(decon_root, "DeconCurves_Manifest.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"[INFO] Exported {len(decon_manifest_rows)} per-spectrum deconvolution curve CSVs -> {decon_root}")

    if ramp_keys:
        ramp_long = pd.concat([all_quant[k] for k in ramp_keys], ignore_index=True)
        ramp_long.rename(columns={"NumVal": "Temp_C"}, inplace=True)
        ramp_long.to_csv(os.path.join(args.out_dir, "Step3_Ramp_AllMetrics_Long.csv"), index=False, encoding="utf-8-sig")

        all_temps = sorted({round(float(t), 3) for k in ramp_keys for t in all_quant[k]["NumVal"].dropna()})
        wide_rows = []
        for tc in all_temps:
            row = {"Temp_C": tc}
            for k in ramp_keys:
                prefix = k[:-5]
                sub = all_quant[k]
                hit = sub[np.isclose(sub["NumVal"], tc)]
                for mcol in ("R_decon_1596_1566", "dR_decon_1596_1566", "R_1590_1568", "dR_1590_1568", "R_1600_1568", "dR_1600_1568"):
                    row[f"{prefix}_{mcol}"] = round(float(hit.iloc[0][mcol]), 5) if len(hit) > 0 else np.nan
            wide_rows.append(row)
        pd.DataFrame(wide_rows).to_csv(
            os.path.join(args.out_dir, "Step3_Ramp_40-120C_ThreeMetrics_Origin.csv"), index=False, encoding="utf-8-sig"
        )

    aligned_end_idx = {}
    if hold_keys:
        hold_frames = []
        aligned_map = {}
        for k in hold_keys:
            qh = all_quant[k].copy()
            raw_t = qh["NumVal"].values.astype(float)
            t_rel = raw_t - raw_t[0]
            qh.insert(2, "Raw_Time_min", np.round(raw_t, 3))
            qh.insert(3, "Hold_Time_min", np.round(t_rel, 3))
            qh.drop(columns=["NumVal"], inplace=True)
            hold_frames.append(qh)

            m_align = qh["Hold_Time_min"] <= args.max_hold_min + 1e-6
            end_i = int(np.where(m_align.values)[0][-1])
            aligned_end_idx[k] = end_i
            aligned_sub = qh[m_align].reset_index(drop=True)
            aligned_map[k[:-5]] = aligned_sub

            # Save direct shortcut CSV matching Fig_SI_Step3_Deconvolution_Verification panel
            cname_e, x_e, y_e, yhat_e, res_e, c_e, _, _ = all_fit_records[k][end_i]
            build_decon_curve_df(x_e, y_e, yhat_e, res_e, c_e).to_csv(
                os.path.join(decon_root, f"SI_AlignedEndpoint_{k}_decon.csv"), index=False, encoding="utf-8-sig"
            )

            print(
                f"  {k:24s}: fit_R2=[{qh['Decon_Fit_R2'].min():.4f},{qh['Decon_Fit_R2'].max():.4f}] "
                f"(RMSE<={qh['Decon_Fit_RMSE'].max():.5f}) | "
                f"R_decon: {qh['R_decon_1596_1566'].iloc[0]:.3f}->{aligned_sub['R_decon_1596_1566'].iloc[-1]:.3f} | "
                f"R1590: {qh['R_1590_1568'].iloc[0]:.3f}->{aligned_sub['R_1590_1568'].iloc[-1]:.3f} | "
                f"R1600: {qh['R_1600_1568'].iloc[0]:.3f}->{aligned_sub['R_1600_1568'].iloc[-1]:.3f}"
            )

        hold_long = pd.concat(hold_frames, ignore_index=True)
        hold_long.to_csv(
            os.path.join(args.out_dir, "Step3_Isothermal_120C_AllMetrics_Long.csv"), index=False, encoding="utf-8-sig"
        )

        max_n = max(len(v) for v in aligned_map.values())
        w_cols = {}
        for pref, sub in aligned_map.items():
            w_cols[f"{pref}_Time_min"] = list(sub["Hold_Time_min"]) + [np.nan] * (max_n - len(sub))
            for mcol in (
                "R_decon_1596_1566", "dR_decon_1596_1566",
                "R_1590_1568", "dR_1590_1568",
                "R_1600_1568", "dR_1600_1568",
                "Decon_Fit_R2", "Decon_Fit_RMSE",
            ):
                w_cols[f"{pref}_{mcol}"] = list(np.round(sub[mcol], 5)) + [np.nan] * (max_n - len(sub))
        pd.DataFrame(w_cols).to_csv(
            os.path.join(args.out_dir, "Step3_Isothermal_Aligned_ThreeMetrics_Origin.csv"), index=False, encoding="utf-8-sig"
        )

    # ==========================================================================
    # Verification Plots: Step 2 Vapor Audit, SI Endpoint Decon, & Full Grid QC
    # ==========================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    t_raw = dfs[tmpl_key].iloc[:, -1].values
    t_cln = clean_dfs[tmpl_key].iloc[:, -1].values
    axes[0, 0].plot(wn, t_raw, color="#ff9896", lw=1.0, label=f"{tmpl_key} last (Step 1A raw)")
    axes[0, 0].plot(wn, t_cln, color="#d62728", lw=1.6, label=f"{tmpl_key} last (Step 2 cleaned)")
    axes[0, 0].set_xlim(1780, 1455)
    axes[0, 0].set_xlabel("Wavenumber (cm⁻¹)"); axes[0, 0].set_ylabel("Absorbance")
    axes[0, 0].set_title(f"(a) {tmpl_key}: Before vs. After H₂O(g) Subtraction")
    axes[0, 0].legend(fontsize=8.2); axes[0, 0].grid(alpha=0.25)

    non_tmpl_keys = [k for k in hold_keys if k != tmpl_key][:2]
    for idx_b, k in enumerate(non_tmpl_keys):
        col = PALETTE[idx_b % len(PALETTE)]
        off = 0.020 * idx_b
        axes[0, 1].plot(wn, dfs[k].iloc[:, -1].values + off, color=col, alpha=0.45, lw=1.6, label=f"{k} raw")
        axes[0, 1].plot(wn, clean_dfs[k].iloc[:, -1].values + off, color=col, ls="--", lw=1.3, label=f"{k} Step 2")
    axes[0, 1].set_xlim(1780, 1455)
    axes[0, 1].set_xlabel("Wavenumber (cm⁻¹)"); axes[0, 1].set_ylabel("Absorbance")
    axes[0, 1].set_title("(b) Low-Vapor / Control Non-Distortion Check")
    if non_tmpl_keys:
        axes[0, 1].legend(fontsize=8.0)
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(wn, W, color="#2ca02c", lw=1.1, label="H₂O(g) ν₂ template W(ν)")
    axes[1, 0].axvspan(1700, 1800, color="#fff3cd", alpha=0.65, label="Anchor window (1700–1800 cm⁻¹)")
    axes[1, 0].axhline(0, color="k", ls="--", lw=0.7)
    axes[1, 0].set_xlim(1800, 1340)
    axes[1, 0].set_xlabel("Wavenumber (cm⁻¹)"); axes[1, 0].set_ylabel("Normalized W(ν)")
    axes[1, 0].set_title("(c) Gas-Phase H₂O(g) Template W(ν)")
    axes[1, 0].legend(fontsize=8.0, loc="lower right"); axes[1, 0].grid(alpha=0.25)

    plot_audit_keys = hold_keys if hold_keys else list(dfs.keys())
    for idx_k, k in enumerate(plot_audit_keys):
        sub = audit2_df[audit2_df["Dataset"] == k]
        axes[1, 1].plot(
            range(len(sub)), sub["corr_1700_1800"],
            marker=MARKERS[idx_k % len(MARKERS)], color=PALETTE[idx_k % len(PALETTE)],
            ms=3.8, lw=1.5, label=k
        )
    axes[1, 1].axhline(args.corr_gate, color="gray", ls="--", lw=0.9, label=f"Gate ±{args.corr_gate:.2f}")
    axes[1, 1].axhline(-args.corr_gate, color="gray", ls="--", lw=0.9)
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_xlabel("Spectrum Index"); axes[1, 1].set_ylabel("Pearson r (1700–1800 cm⁻¹)")
    axes[1, 1].set_title("(d) Step 2 Correlation Gate Audit (Signed Humidity Check)")
    axes[1, 1].legend(fontsize=8.0); axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "Fig_Step2_H2O_Vapor_Verification.png"), dpi=240, bbox_inches="tight")
    plt.close()

    comp_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    if hold_keys:
        fig = plt.figure(figsize=(5.0 * len(hold_keys), 5.3))
        outer = GridSpec(2, len(hold_keys), figure=fig, height_ratios=[3.3, 1.05], hspace=0.08, wspace=0.16)
        ax_top_0, ax_res_0 = None, None
        for col_i, k in enumerate(hold_keys):
            idx_pick = aligned_end_idx[k]
            cname, x_s, y_s, y_hat, resid, c, r2, rmse = all_fit_records[k][idx_pick]
            ax_top = fig.add_subplot(outer[0, col_i], sharey=ax_top_0)
            ax_res = fig.add_subplot(outer[1, col_i], sharex=ax_top, sharey=ax_res_0)
            if col_i == 0:
                ax_top_0, ax_res_0 = ax_top, ax_res

            ax_top.plot(x_s, y_s, "k-", lw=1.6, label="Experimental S_clean(ν)")
            ax_top.plot(x_s, y_hat, "r--", lw=1.4, label="6-Gaussian Sum Fit Ŝ(ν)")
            for p_i, (mu, sig, plbl, _) in enumerate(SHARED_BASIS_PEAKS):
                comp = c[p_i] * np.exp(-0.5 * ((x_s - mu) / sig) ** 2)
                ax_top.fill_between(x_s, c[-1], c[-1] + comp, color=comp_colors[p_i], alpha=0.22, label=plbl)
            ax_top.axhline(c[-1], color="#777777", ls=":", lw=0.8)

            t_lbl = parse_num(cname)
            t0_val = parse_num(clean_dfs[k].columns[1])
            t_rel_end = (t_lbl - t0_val) if (np.isfinite(t_lbl) and np.isfinite(t0_val)) else np.nan
            t_str = f"t = {t_rel_end:.1f} min" if np.isfinite(t_rel_end) else "end"
            ax_top.set_title(f"{k} ({t_str})\nR² = {r2:.4f}, RMSE = {rmse:.5f}", fontsize=9.8)
            ax_top.grid(alpha=0.25)
            plt.setp(ax_top.get_xticklabels(), visible=False)

            ax_res.plot(x_s, resid * 1e3, color="#444444", lw=1.05)
            ax_res.axhline(0.0, color="#d62728", ls="--", lw=0.8)
            ax_res.set_xlim(DECON_WIN[1], DECON_WIN[0])
            ax_res.set_ylim(-4.5, 4.5)
            ax_res.set_xlabel("Wavenumber (cm⁻¹)")
            ax_res.grid(alpha=0.25)

            if col_i == 0:
                ax_top.set_ylabel("Absorbance (a.u.)")
                ax_res.set_ylabel("Residual\n(×10⁻³)")
            else:
                plt.setp(ax_top.get_yticklabels(), visible=False)
                plt.setp(ax_res.get_yticklabels(), visible=False)

            if col_i == len(hold_keys) - 1:
                ax_top.legend(fontsize=7.2, loc="upper left", frameon=True, framealpha=0.90)

        fig.savefig(os.path.join(args.out_dir, "Fig_SI_Step3_Deconvolution_Verification.png"), dpi=260, bbox_inches="tight")
        fig.savefig(os.path.join(args.out_dir, "Fig_SI_Step3_Deconvolution_Verification.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Full-dataset multi-panel deconvolution QC grids (every single spectrum plotted!)
    for k in ordered_all_keys:
        frec = all_fit_records[k]
        n_sp = len(frec)
        ncols = min(5, n_sp)
        nrows = math.ceil(n_sp / ncols)
        fig, axes_g = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.55 * nrows), sharex=True, sharey=True)
        axes_flat = np.atleast_1d(axes_g).ravel()
        first_num = parse_num(clean_dfs[k].columns[1]) if clean_dfs[k].shape[1] > 1 else np.nan
        for idx_s, (cname, x_s, y_s, y_hat, resid, c, r2, rmse) in enumerate(frec):
            ax_g = axes_flat[idx_s]
            ax_g.plot(x_s, y_s, "k-", lw=1.15, label="Exp" if idx_s == 0 else None)
            ax_g.plot(x_s, y_hat, "r--", lw=1.0, label="Sum Fit" if idx_s == 0 else None)
            for p_i, (mu, sig, plbl, _) in enumerate(SHARED_BASIS_PEAKS):
                comp = c[p_i] * np.exp(-0.5 * ((x_s - mu) / sig) ** 2)
                ax_g.fill_between(x_s, c[-1], c[-1] + comp, color=comp_colors[p_i], alpha=0.20, label=plbl if idx_s == 0 else None)
            ax_g.axhline(c[-1], color="#888888", ls=":", lw=0.6)
            ax_g.set_xlim(DECON_WIN[1], DECON_WIN[0])
            _, axis_val, axis_unit = format_axis_tag(k, cname, parse_num(cname), first_num)
            hdr = f"{axis_val:.2f} min" if axis_unit == "min" else (f"{axis_val:g} °C" if axis_unit == "C" else cname)
            ax_g.set_title(f"#{idx_s:02d} {hdr} | R²={r2:.3f}", fontsize=8.0)
            ax_g.grid(alpha=0.20)
        for rem_i in range(n_sp, len(axes_flat)):
            axes_flat[rem_i].set_visible(False)
        fig.suptitle(f"{k}: All-Spectra 6-Gaussian Shared-Basis Deconvolution QC ({n_sp} spectra)", fontsize=11.0, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(decon_root, f"Fig_AllSpectra_DeconGrid_{k}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[DONE] Step 2 & Step 3 outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
