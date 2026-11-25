#!/usr/bin/env python3
"""
Step 4 (Reviewer-Hardened & Sample-Agnostic):
  Nature-style 4-Panel In-Situ FTIR Figures + Origin Matrices:
    - Clean catalytic diagnostic window tightened to [1475.0, 1750.0] cm^-1
      (eliminates non-catalytic 1432 cm^-1 shoulder spillover while keeping 1487 cm^-1).
    - Unified symmetric diverging color scale [-vmax, +vmax] via 99.5th percentile + 0.0005.
    - Generates BOTH 120 °C Isothermal Fig. 4a-d (D1 stacked unified-y & D2 overlay)
      AND 40 -> 120 °C Ramp 4-panel figure + all Origin 2D difference matrices & OLS linear fit tables.
    - Auto-discovers sample prefixes (with Blank first) or accepts --samples / --labels via CLI.
    - Supports --metric {decon, r1590, r1600, all} (default: decon).

Usage:
  python3 scripts/step4_isothermal_fig4_template.py --in-dir ./output/step2_3 --out-dir ./output/step4 --metric all
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress

DIAG_WIN = (1475.0, 1750.0)
PALETTE_D = ["#333333", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
MARKERS_D = ["o", "s", "^", "D", "v", "P"]

METRIC_META = {
    "decon": ("R_decon_1596_1566", "dR_decon_1596_1566", "Deconvolved Area Ratio I(1596)/I(1566)", "Δ[I(1596)/I(1566)]"),
    "r1590": ("R_1590_1568",       "dR_1590_1568",       "Ratio A(1590) / A(1568)",                "Δ[A(1590)/A(1568)]"),
    "r1600": ("R_1600_1568",       "dR_1600_1568",       "Ratio A(1600) / A(1568)",                "Δ[A(1600)/A(1568)]"),
}


def parse_num(s):
    m = re.search(r"([0-9]+\.?[0-9]*)", str(s))
    return float(m.group(1)) if m else np.nan


def discover_hold_samples(in_dir, requested_samples=None):
    if requested_samples:
        return list(requested_samples)
    paths = sorted(glob.glob(os.path.join(in_dir, "*_Hold_1000-1800_bl1A_wvclean.csv")))
    prefixes = [os.path.basename(p).replace("_Hold_1000-1800_bl1A_wvclean.csv", "") for p in paths]
    blanks = sorted([p for p in prefixes if p.lower() == "blank"])
    others = sorted([p for p in prefixes if p.lower() != "blank"])
    ordered = blanks + others
    if not ordered:
        raise FileNotFoundError(f"No *_Hold_1000-1800_bl1A_wvclean.csv files found in {in_dir}")
    return ordered


def run_for_metric(metric, args, sample_keys, hold_long, ramp_long, export_diff_matrices=True):
    wn_win = (args.wn_min, args.wn_max)
    r_col, dr_col, y_lbl_abs, y_lbl_del = METRIC_META[metric]

    # ==========================================================================
    # PART A: 120 °C Isothermal Hold (Fig. 4a-d, D1 Stacked & D2 Overlay)
    # ==========================================================================
    loaded_hold = []
    fit_rows = []
    for prefix, disp in sample_keys:
        key = f"{prefix}_Hold"
        df = pd.read_csv(os.path.join(args.in_dir, f"{key}_1000-1800_bl1A_wvclean.csv"), encoding="utf-8-sig")
        w_all = df.iloc[:, 0].values.astype(float)
        cols = df.columns[1:]
        raw_t = np.array([parse_num(c) for c in cols], dtype=float)
        t_rel = raw_t - raw_t[0]
        m_t = t_rel <= (args.max_hold_min + 1e-6)
        t_rel = t_rel[m_t]
        mat = df.iloc[:, 1:].values[:, m_t].astype(float)
        diff = mat - mat[:, 0:1]

        m_w = (w_all >= wn_win[0]) & (w_all <= wn_win[1])
        w_d = w_all[m_w]
        diff_d = diff[m_w, :]

        sub_q = hold_long[hold_long["Dataset"] == key].reset_index(drop=True)
        sub_q = sub_q[sub_q["Hold_Time_min"] <= args.max_hold_min + 1e-6].reset_index(drop=True)
        ratios = sub_q[r_col].values.astype(float)
        d_ratios = sub_q[dr_col].values.astype(float)

        res_R = linregress(t_rel, ratios)
        res_dR = linregress(t_rel, d_ratios)

        loaded_hold.append({
            "prefix": prefix, "disp": disp, "wn": w_d, "t": t_rel,
            "diff": diff_d, "ratios": ratios, "d_ratios": d_ratios,
            "fit_R": res_R, "fit_dR": res_dR,
        })
        fit_rows.append({
            "Sample": disp, "Metric": metric, "n_spectra": len(t_rel),
            "t_max_min": round(float(t_rel[-1]), 3),
            "R_t0": round(float(ratios[0]), 5), "R_tend": round(float(ratios[-1]), 5),
            "Delta_R": round(float(d_ratios[-1]), 5),
            "Slope_k_per_min": round(res_R.slope, 6), "Slope_SE": round(res_R.stderr, 6),
            "Intercept": round(res_R.intercept, 5), "R2": round(res_R.rvalue ** 2, 5),
            "p_value": f"{res_R.pvalue:.3e}",
        })

        if export_diff_matrices:
            mdf = pd.DataFrame(np.round(diff_d, 7), columns=[f"{tv:.2f} min" for tv in t_rel])
            mdf.insert(0, "Wavenumber (cm-1)", np.round(w_d, 4))
            mdf.to_csv(
                os.path.join(args.out_dir, f"Origin_Isothermal_DiffMatrix_{prefix}.csv"), index=False, encoding="utf-8-sig"
            )

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(
        os.path.join(args.out_dir, f"Origin_Isothermal_PanelD_LinearFit_{metric}.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"=== Step 4 Isothermal Linear Fit Summary (metric={metric}) ===")
    print(fit_df.to_string(index=False))

    vmax_hold = round(max(float(np.percentile(np.abs(it["diff"]), 99.5)) for it in loaded_hold) + 0.0005, 3)

    if len(loaded_hold) == 3:
        def render_isothermal(mode, out_stem):
            plt.rcParams.update({
                "font.family": "DejaVu Sans", "font.size": 9.5,
                "axes.titlesize": 10.5, "axes.titleweight": "bold",
            })
            fig = plt.figure(figsize=(15.2, 10.8))
            outer = GridSpec(2, 2, figure=fig, left=0.065, right=0.96, top=0.94, bottom=0.07, wspace=0.40, hspace=0.28)
            tags, slots = ["a", "b", "c"], [(0, 0), (0, 1), (1, 0)]
            line_colors = ["#222222", "#6baed6", "#3182bd", "#e6550d", "#b2182b"]

            im_ref = None
            ax_bots = []
            for idx, it in enumerate(loaded_hold):
                r, c = slots[idx]
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.05, 2.15], hspace=0.06)
                ax_top = fig.add_subplot(inner[0])
                ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
                ax_bots.append(ax_bot)

                w, t, D = it["wn"], it["t"], it["diff"]
                pick = np.unique(np.linspace(0, len(t) - 1, 5).round().astype(int))
                for k_i, j in enumerate(pick):
                    ax_top.plot(w, D[:, j], lw=1.25, color=line_colors[k_i % len(line_colors)], label=f"{t[j]:.1f} min")
                ax_top.axhline(0, color="#666666", ls=":", lw=0.75)
                for xv in [1634, 1600, 1568, 1487]:
                    ax_top.axvline(xv, color="#555555", ls="--", lw=0.75, alpha=0.65)
                    ax_bot.axvline(xv, color="#222222", ls="--", lw=0.6, alpha=0.40)
                    ax_top.text(
                        xv, -vmax_hold * 1.36, str(xv), ha="center", va="bottom", fontsize=7.0, color="#444444",
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.80)
                    )

                ax_top.set_ylim(-vmax_hold * 1.42, vmax_hold * 1.25)
                ax_top.set_ylabel("ΔAbs (a.u.)")
                ax_top.set_title(f"{tags[idx]}   {it['disp']} (120 °C Isothermal)", loc="left", pad=6)
                ax_top.legend(ncol=3, fontsize=7.3, frameon=False, loc="upper left", handlelength=1.4, columnspacing=0.9)
                ax_top.spines["top"].set_visible(False)
                ax_top.spines["right"].set_visible(False)
                plt.setp(ax_top.get_xticklabels(), visible=False)

                im_ref = ax_bot.pcolormesh(w, t, D.T, cmap="RdBu_r", vmin=-vmax_hold, vmax=vmax_hold, shading="gouraud")
                ax_bot.set_xlim(wn_win[1], wn_win[0])
                ax_bot.set_ylim(t[0], t[-1])
                ax_bot.set_xlabel("Wavenumber (cm⁻¹)")
                ax_bot.set_ylabel("Time (min)")

            pos_c = ax_bots[2].get_position()
            cbar_ax = fig.add_axes([pos_c.x1 + 0.010, pos_c.y0, 0.013, pos_c.height * 1.48])
            cbar = fig.colorbar(im_ref, cax=cbar_ax)
            cbar.set_label("ΔAbs (a.u.)", fontsize=8.8, labelpad=3)
            cbar.ax.tick_params(labelsize=8.0)

            if mode == "D1_stacked":
                inner_d = outer[1, 1].subgridspec(3, 1, hspace=0.16)
                y_all = np.concatenate([it["ratios"] for it in loaded_hold])
                yrange = max(float(y_all.max() - y_all.min()), 0.1)
                ymin_d = float(y_all.min() - 0.10 * yrange)
                ymax_d = float(y_all.max() + 0.24 * yrange)
                for idx, it in enumerate(loaded_hold):
                    ax_d = fig.add_subplot(inner_d[idx])
                    res = it["fit_R"]
                    col = PALETTE_D[idx % len(PALETTE_D)]
                    mk = MARKERS_D[idx % len(MARKERS_D)]
                    ax_d.scatter(it["t"], it["ratios"], s=22, marker=mk, color=col, alpha=0.85, zorder=3)
                    ax_d.plot(it["t"], res.intercept + res.slope * it["t"], color=col, lw=1.8, zorder=4)
                    ax_d.set_ylim(ymin_d, ymax_d)
                    ax_d.set_xlim(-0.3, args.max_hold_min + 0.3)
                    ax_d.text(
                        0.03, 0.80,
                        f"{it['disp']}   |   k = {res.slope:+.4f} ± {res.stderr:.4f} min⁻¹   (R² = {res.rvalue**2:.3f})",
                        transform=ax_d.transAxes, fontsize=8.2, fontweight="bold", color=col,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85)
                    )
                    ax_d.spines["top"].set_visible(False)
                    ax_d.spines["right"].set_visible(False)
                    ax_d.grid(alpha=0.25)
                    if idx == 0:
                        ax_d.set_title(f"d   Apparent MEA Regeneration Rate Slope k ({metric})", loc="left", pad=6)
                    if idx == 1:
                        ax_d.set_ylabel(y_lbl_abs)
                    if idx < 2:
                        plt.setp(ax_d.get_xticklabels(), visible=False)
                    else:
                        ax_d.set_xlabel("120 °C Isothermal Hold Time (min)")
            else:
                ax_d = fig.add_subplot(outer[1, 1])
                for idx, it in enumerate(loaded_hold):
                    res = it["fit_dR"]
                    col = PALETTE_D[idx % len(PALETTE_D)]
                    mk = MARKERS_D[idx % len(MARKERS_D)]
                    ax_d.scatter(it["t"], it["d_ratios"], s=26, marker=mk, color=col, alpha=0.85, zorder=3)
                    ax_d.plot(
                        it["t"], res.intercept + res.slope * it["t"], color=col, lw=2.0, zorder=4,
                        label=f"{it['disp']}: k = {res.slope:+.4f} ± {res.stderr:.4f} min⁻¹ (R²={res.rvalue**2:.3f})"
                    )
                ax_d.axhline(0, color="#666666", ls=":", lw=0.8)
                ax_d.set_xlim(-0.3, args.max_hold_min + 0.3)
                ax_d.set_title(f"d   Net Increment {y_lbl_del} & Rate Slope k", loc="left", pad=6)
                ax_d.set_xlabel("120 °C Isothermal Hold Time (min)")
                ax_d.set_ylabel(f"{y_lbl_del} vs. t = 0 min")
                ax_d.legend(fontsize=8.0, frameon=True, framealpha=0.92, loc="upper left")
                ax_d.spines["top"].set_visible(False)
                ax_d.spines["right"].set_visible(False)
                ax_d.grid(alpha=0.25)

            fig.savefig(os.path.join(args.out_dir, f"{out_stem}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(args.out_dir, f"{out_stem}.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved {out_stem}.png / .pdf")

        render_isothermal("D1_stacked", f"Fig4_Isothermal_D1_Stacked_{metric}")
        render_isothermal("D2_overlay", f"Fig4_Isothermal_D2_Overlay_{metric}")

    # ==========================================================================
    # PART B: Temperature Ramp 40 -> 120 °C 4-Panel Figure + Origin Matrices
    # ==========================================================================
    if ramp_long is not None:
        loaded_ramp = []
        for prefix, disp in sample_keys:
            key = f"{prefix}_Ramp"
            fp = os.path.join(args.in_dir, f"{key}_1000-1800_bl1A_wvclean.csv")
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp, encoding="utf-8-sig")
            w_all = df.iloc[:, 0].values.astype(float)
            temps = np.array([parse_num(c) for c in df.columns[1:]], dtype=float)
            mat = df.iloc[:, 1:].values.astype(float)
            diff = mat - mat[:, 0:1]
            m_w = (w_all >= wn_win[0]) & (w_all <= wn_win[1])
            sub_q = ramp_long[ramp_long["Dataset"] == key].reset_index(drop=True)
            loaded_ramp.append({
                "prefix": prefix, "disp": disp, "wn": w_all[m_w], "T": temps,
                "diff": diff[m_w, :], "d_ratios": sub_q[dr_col].values.astype(float),
            })
            if export_diff_matrices:
                mdf = pd.DataFrame(np.round(diff[m_w, :], 7), columns=[f"{int(tv)} C" for tv in temps])
                mdf.insert(0, "Wavenumber (cm-1)", np.round(w_all[m_w], 4))
                mdf.to_csv(
                    os.path.join(args.out_dir, f"Origin_Ramp_DiffMatrix_{prefix}.csv"), index=False, encoding="utf-8-sig"
                )

        if len(loaded_ramp) == 3:
            vmax_rmp = round(max(float(np.percentile(np.abs(it["diff"]), 99.5)) for it in loaded_ramp) + 0.0005, 3)
            fig = plt.figure(figsize=(15.2, 10.8))
            outer = GridSpec(2, 2, figure=fig, left=0.065, right=0.96, top=0.94, bottom=0.07, wspace=0.40, hspace=0.28)
            tags, slots = ["a", "b", "c"], [(0, 0), (0, 1), (1, 0)]
            line_colors = ["#222222", "#6baed6", "#3182bd", "#e6550d", "#b2182b"]
            im_ref = None
            ax_bots = []
            for idx, it in enumerate(loaded_ramp):
                r, c = slots[idx]
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.05, 2.15], hspace=0.06)
                ax_top = fig.add_subplot(inner[0])
                ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
                ax_bots.append(ax_bot)
                w, T, D = it["wn"], it["T"], it["diff"]
                pick = [int(np.argmin(np.abs(T - tv))) for tv in [40, 60, 80, 100, 120]]
                for k_i, j in enumerate(pick):
                    ax_top.plot(w, D[:, j], lw=1.25, color=line_colors[k_i], label=f"{int(T[j])} °C")
                ax_top.axhline(0, color="#666666", ls=":", lw=0.75)
                for xv in [1634, 1600, 1568, 1487]:
                    ax_top.axvline(xv, color="#555555", ls="--", lw=0.75, alpha=0.65)
                    ax_bot.axvline(xv, color="#222222", ls="--", lw=0.6, alpha=0.40)
                    ax_top.text(
                        xv, -vmax_rmp * 1.36, str(xv), ha="center", va="bottom", fontsize=7.0, color="#444444",
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.80)
                    )
                ax_top.set_ylim(-vmax_rmp * 1.42, vmax_rmp * 1.35)
                ax_top.set_ylabel("ΔAbs vs. 40 °C")
                ax_top.set_title(f"{tags[idx]}   {it['disp']} (40 → 120 °C Ramp)", loc="left", pad=6)
                ax_top.legend(ncol=3, fontsize=7.3, frameon=False, loc="upper left", handlelength=1.4, columnspacing=0.9)
                ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
                plt.setp(ax_top.get_xticklabels(), visible=False)
                im_ref = ax_bot.pcolormesh(w, T, D.T, cmap="RdBu_r", vmin=-vmax_rmp, vmax=vmax_rmp, shading="gouraud")
                ax_bot.set_xlim(wn_win[1], wn_win[0]); ax_bot.set_ylim(T[0], T[-1])
                ax_bot.set_xlabel("Wavenumber (cm⁻¹)"); ax_bot.set_ylabel("Temperature (°C)")

            pos_c = ax_bots[2].get_position()
            cbar_ax = fig.add_axes([pos_c.x1 + 0.010, pos_c.y0, 0.013, pos_c.height * 1.48])
            cbar = fig.colorbar(im_ref, cax=cbar_ax)
            cbar.set_label("ΔAbs vs. 40 °C (a.u.)", fontsize=8.8, labelpad=3)
            cbar.ax.tick_params(labelsize=8.0)

            ax_d = fig.add_subplot(outer[1, 1])
            for idx, it in enumerate(loaded_ramp):
                col = PALETTE_D[idx % len(PALETTE_D)]
                mk = MARKERS_D[idx % len(MARKERS_D)]
                ax_d.plot(
                    it["T"], it["d_ratios"], marker=mk, color=col, ms=5.0, lw=2.0,
                    label=f"{it['disp']} (Δ₁₂₀ = {it['d_ratios'][-1]:+.3f})"
                )
            ax_d.axhline(0, color="#666666", ls=":", lw=0.8)
            ax_d.set_xlim(37, 123)
            ax_d.set_title(f"d   Temperature-Programmed {y_lbl_del}", loc="left", pad=6)
            ax_d.set_xlabel("Temperature (°C)"); ax_d.set_ylabel(f"{y_lbl_del} vs. 40 °C")
            ax_d.legend(fontsize=8.3, frameon=True, framealpha=0.92, loc="upper left")
            ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
            ax_d.grid(alpha=0.25)

            stem = f"Fig_Ramp_40-120C_4Panel_{metric}"
            fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(args.out_dir, f"{stem}.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved {stem}.png / .pdf")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Generate Nature-style 4-panel FTIR figures & Origin matrices.")
    parser.add_argument("--in-dir", default="./output/step2_3", help="Directory containing Step 2 *_wvclean.csv & Step 3 ratio CSVs.")
    parser.add_argument("--out-dir", default="./output/step4", help="Output directory for figures and Origin matrices.")
    parser.add_argument(
        "--metric", choices=["decon", "r1590", "r1600", "all"], default="decon",
        help="Ratio metric for Panel d (default: decon; use 'all' to generate all three)."
    )
    parser.add_argument("--max-hold-min", type=float, default=10.6, help="Aligned 120 °C isothermal hold window (min).")
    parser.add_argument("--wn-min", type=float, default=DIAG_WIN[0], help="Lower diagnostic wavenumber (default: 1475 cm^-1).")
    parser.add_argument("--wn-max", type=float, default=DIAG_WIN[1], help="Upper diagnostic wavenumber (default: 1750 cm^-1).")
    parser.add_argument("--samples", nargs="+", default=None, help="Optional ordered sample prefixes (default: auto-discover with Blank first).")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional display labels matching --samples / auto-discovered samples.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hold_long = pd.read_csv(os.path.join(args.in_dir, "Step3_Isothermal_120C_AllMetrics_Long.csv"), encoding="utf-8-sig")
    ramp_long_path = os.path.join(args.in_dir, "Step3_Ramp_AllMetrics_Long.csv")
    ramp_long = pd.read_csv(ramp_long_path, encoding="utf-8-sig") if os.path.exists(ramp_long_path) else None

    samples = discover_hold_samples(args.in_dir, args.samples)
    if args.labels:
        if len(args.labels) != len(samples):
            raise ValueError(f"--labels length ({len(args.labels)}) must match number of samples ({len(samples)}): {samples}")
        labels = list(args.labels)
    else:
        labels = list(samples)
    sample_keys = list(zip(samples, labels))

    metrics_to_run = ["decon", "r1590", "r1600"] if args.metric == "all" else [args.metric]
    for i, m in enumerate(metrics_to_run):
        run_for_metric(m, args, sample_keys, hold_long, ramp_long, export_diff_matrices=(i == 0))

    print(f"\n[DONE] Step 4 outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
