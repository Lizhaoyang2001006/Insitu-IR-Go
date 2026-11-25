#!/usr/bin/env python3
"""
Step 0 & Step 1A: Raw FTIR Audit, Unified Truncation (1000-1800 cm^-1),
and Horizontal Median Baseline Zeroing (1750-1800 cm^-1).

Mathematical Specification (1:1 matched with INSITU_FTIR_HANDOFF_GUIDE.md):
  1. Wavenumber truncation:
       Omega_1A = { nu_i | wn_min <= nu_i <= wn_max }, default [999.9, 1800.3] cm^-1.
  2. Step 0 automated audit gates:
       - Gate 0.1: Wavenumber column has 0 NaN/Inf and is strictly increasing (Delta_nu > 0).
       - Gate 0.2: Stray wavenumber columns (min > 900 and max > 3500) are automatically excluded.
       - Gate 0.3: Absorbance columns have 0 NaN/Inf on Omega_1A; duplicate spectra (allclose)
                   against any preceding column in the file are flagged.
       - Gate 0.4: 120 °C Ramp-end vs. Hold-t0 continuity RMS on [1475.0, 1750.0] cm^-1.
  3. Horizontal median baseline zeroing on absorption-free anchor Omega_bl = [1750.0, 1800.0] cm^-1:
       b_j = median_{nu in Omega_bl} S_j(nu)
       S_{j,1A}(nu) = S_j(nu) - b_j

Path-agnostic & sample-agnostic CLI tool: supports CSV and XLSX files with 1-row or multi-row headers.
Usage:
  python3 scripts/step1A_baseline_truncate.py --input-dir ./raw_data --out-dir ./output/step1A
"""
import argparse
import csv
import glob
import os
import re
import numpy as np
import pandas as pd

WN_MIN, WN_MAX = 999.9, 1800.3
BL_MIN, BL_MAX = 1750.0, 1800.0
DIAG_MIN, DIAG_MAX = 1475.0, 1750.0


def detect_encoding(fp):
    with open(fp, "rb") as f:
        raw = f.read(8192)
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def load_raw_table(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(fp, sheet_name=0, header=None)
        headers = [str(x).strip() if pd.notna(x) else f"col_{i}" for i, x in enumerate(raw.iloc[0, :])]
        start_row = None
        for r in range(min(10, len(raw))):
            try:
                val = float(raw.iloc[r, 0])
                if np.isfinite(val):
                    start_row = r
                    break
            except Exception:
                pass
        if start_row is None:
            raise ValueError(f"Cannot detect numeric data start row in {fp}")
        df = raw.iloc[start_row:, :].reset_index(drop=True)
        return headers, df, start_row

    enc = detect_encoding(fp)
    with open(fp, "r", encoding=enc, errors="replace") as f:
        first_lines = [f.readline().rstrip("\r\n") for _ in range(10)]
    headers = [c.strip() for c in next(csv.reader([first_lines[0]]))]
    start_row = None
    for i, ln in enumerate(first_lines):
        parts = next(csv.reader([ln]))
        if parts and parts[0].strip() != "":
            try:
                val = float(parts[0])
                if np.isfinite(val):
                    start_row = i
                    break
            except Exception:
                pass
    if start_row is None:
        raise ValueError(f"Cannot detect numeric data start row in {fp}")
    df = pd.read_csv(fp, encoding=enc, skiprows=start_row, header=None)
    return headers, df, start_row


def infer_canonical_key(filename):
    """Infer '<Sample>_<Hold|Ramp|Data>' generically from filename without hardcoded sample names."""
    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    low = stem.lower()
    if any(k in low for k in ("120", "time", "hold", "isothermal", "稳定", "恒温")):
        mode = "Hold"
    elif any(k in low for k in ("temp", "ramp", "升温")):
        mode = "Ramp"
    else:
        mode = "Data"

    sample = re.sub(
        r"[-_\s]*(?:120[-_\s]*time|120|time|hold|isothermal|temperture|temperature|temp|ramp|稳定|恒温|升温)+$",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip("-_ ")
    if not sample:
        sample = stem
    if sample.lower() == "blank":
        sample = "Blank"
    return f"{sample}_{mode}"


def discover_input_files(input_dir):
    candidates = sorted(
        glob.glob(os.path.join(input_dir, "*.csv"))
        + glob.glob(os.path.join(input_dir, "*.xlsx"))
    )
    files = [
        f for f in candidates
        if not any(tag in os.path.basename(f) for tag in ("_bl1A", "_wvclean", "Step1A_", "Step2_", "Step3_", "Origin_"))
    ]
    if not files:
        raise FileNotFoundError(f"No raw .csv or .xlsx files found in: {input_dir}")
    return files


def order_sample_prefixes(prefixes):
    unique = sorted(set(prefixes))
    blanks = [p for p in unique if p.lower() == "blank"]
    others = [p for p in unique if p.lower() != "blank"]
    return blanks + others


def main():
    parser = argparse.ArgumentParser(
        description="Step 0 & Step 1A: FTIR audit, 1000-1800 truncation, and 1750-1800 median baseline zeroing."
    )
    parser.add_argument("--input-dir", default="./raw_data", help="Directory containing the raw CSV/XLSX FTIR files.")
    parser.add_argument("--out-dir", default="./output/step1A", help="Output directory for Step 1A CSVs and audit log.")
    parser.add_argument("--wn-min", type=float, default=WN_MIN, help="Lower wavenumber cutoff (cm^-1).")
    parser.add_argument("--wn-max", type=float, default=WN_MAX, help="Upper wavenumber cutoff (cm^-1).")
    parser.add_argument("--bl-min", type=float, default=BL_MIN, help="Horizontal baseline anchor start (cm^-1).")
    parser.add_argument("--bl-max", type=float, default=BL_MAX, help="Horizontal baseline anchor end (cm^-1).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = discover_input_files(args.input_dir)

    audit_rows = []
    manifest_rows = []
    processed_dfs = {}
    used_keys = set()

    print("=" * 88)
    print("STEP 0 & STEP 1A: Raw Audit + Truncation + Horizontal Median Baseline Zeroing")
    print("=" * 88)

    for fp in files:
        fname = os.path.basename(fp)
        key = infer_canonical_key(fname)
        if key in used_keys:
            key = os.path.splitext(fname)[0]
        used_keys.add(key)

        headers, df, start_row = load_raw_table(fp)
        wn_full = pd.to_numeric(df.iloc[:, 0], errors="coerce").values.astype(float)
        n_bad_wn = int((~np.isfinite(wn_full)).sum())
        if n_bad_wn > 0:
            raise ValueError(
                f"[AUDIT FAIL] {fname}: Wavenumber column has {n_bad_wn}/{len(wn_full)} NaN/Inf entries!"
            )
        diffs = np.diff(wn_full)
        if np.any(diffs <= 0):
            raise ValueError(f"[AUDIT FAIL] {fname}: Wavenumber column is not strictly monotonic increasing!")

        m_range = (wn_full >= args.wn_min) & (wn_full <= args.wn_max)
        wn_sub = wn_full[m_range]
        if len(wn_sub) < 100:
            raise ValueError(f"[AUDIT FAIL] {fname}: Only {len(wn_sub)} points in [{args.wn_min}, {args.wn_max}] cm^-1.")
        m_bl = (wn_sub >= args.bl_min) & (wn_sub <= args.bl_max)
        if int(m_bl.sum()) < 10:
            raise ValueError(f"[AUDIT FAIL] {fname}: Insufficient baseline points in [{args.bl_min}, {args.bl_max}] cm^-1.")

        out_dict = {"Wavenumber (cm-1)": np.round(wn_sub, 5)}
        used_names = {"Wavenumber (cm-1)"}
        dropped_cols = []
        dup_data_cols = []
        seen_spectra = []

        for j in range(1, df.shape[1]):
            cname = headers[j] if j < len(headers) else f"col_{j}"
            y_full = pd.to_numeric(df.iloc[:, j], errors="coerce").values.astype(float)

            # Gate 0.2: Guard against accidentally pasted wavenumber column
            if np.nanmin(y_full) > 900.0 and np.nanmax(y_full) > 3500.0:
                dropped_cols.append(f"col {j} ('{cname}')")
                audit_rows.append({
                    "Dataset": key, "SourceFile": fname, "ColIdx": j,
                    "OrigColName": cname, "OutColName": "DROPPED (stray wavenumber col)",
                    "Baseline_Median": np.nan, "Baseline_Mean": np.nan,
                    "Post_Min": np.nan, "Post_Max": np.nan,
                    "Note": "Stray wavenumber column excluded"
                })
                continue

            y_sub = y_full[m_range]
            # Gate 0.3a: Finite absorbance check on truncated domain
            n_bad_y = int((~np.isfinite(y_sub)).sum())
            if n_bad_y > 0:
                raise ValueError(
                    f"[AUDIT FAIL] {fname} col {j} ('{cname}'): {n_bad_y} NaN/Inf absorbance values in [{args.wn_min}, {args.wn_max}] cm^-1!"
                )

            # Gate 0.3b: Duplicate check against all preceding spectra in the same file
            note_parts = []
            for prev_name, prev_y in seen_spectra:
                if np.allclose(y_sub, prev_y, rtol=1e-5, atol=1e-8):
                    dup_data_cols.append(f"col {j} ('{cname}' == '{prev_name}')")
                    note_parts.append(f"WARNING: Identical spectrum to '{prev_name}'")
                    break
            seen_spectra.append((cname, y_sub))

            out_cname = cname
            if out_cname in used_names:
                out_cname = f"{cname}_dup{j}"
                note_parts.append(f"Duplicate header '{cname}' disambiguated")
            used_names.add(out_cname)

            b_med = float(np.median(y_sub[m_bl]))
            b_mean = float(np.mean(y_sub[m_bl]))
            y_corr = y_sub - b_med

            out_dict[out_cname] = np.round(y_corr, 7)
            audit_rows.append({
                "Dataset": key, "SourceFile": fname, "ColIdx": j,
                "OrigColName": cname, "OutColName": out_cname,
                "Baseline_Median": round(b_med, 6),
                "Baseline_Mean": round(b_mean, 6),
                "Post_Min": round(float(np.min(y_corr)), 6),
                "Post_Max": round(float(np.max(y_corr)), 6),
                "Note": "; ".join(note_parts)
            })

        out_df = pd.DataFrame(out_dict)
        out_fn = f"{key}_1000-1800_bl1A.csv"
        out_df.to_csv(os.path.join(args.out_dir, out_fn), index=False, encoding="utf-8-sig")
        processed_dfs[key] = out_df
        manifest_rows.append({
            "Dataset": key,
            "SourceFile": fname,
            "Step1A_File": out_fn,
            "HeaderRows": start_row,
            "WN_First": round(float(wn_sub[0]), 5),
            "WN_Last": round(float(wn_sub[-1]), 5),
            "N_Points": len(wn_sub),
            "N_Baseline_Pts": int(m_bl.sum()),
            "N_Spectra": out_df.shape[1] - 1,
            "DroppedCols": "; ".join(dropped_cols),
            "DuplicateDataWarnings": "; ".join(dup_data_cols),
            "Continuity_RMS_1475_1750": np.nan,
        })
        print(
            f"[OK] {key:24s} ({fname}) -> {out_fn} | hdr={start_row} | pts={len(wn_sub)} "
            f"[{wn_sub[0]:.2f}..{wn_sub[-1]:.2f}] | bl_pts={int(m_bl.sum())} | "
            f"spectra={out_df.shape[1]-1:2d} | dropped={dropped_cols} | dups={dup_data_cols}"
        )

    # Gate 0.4: Cross-check 120 °C Ramp-end vs. Hold-t0 continuity on [1475, 1750] cm^-1
    manifest_df = pd.DataFrame(manifest_rows)
    prefixes = order_sample_prefixes(
        [k[:-5] for k in processed_dfs if k.endswith("_Hold") or k.endswith("_Ramp")]
    )
    if prefixes:
        print("-" * 88)
        print("STEP 0 CONTINUITY AUDIT: Ramp(end, 120 °C) vs. Hold(t = 0 min) on 1475-1750 cm^-1")
    for sample in prefixes:
        rk, hk = f"{sample}_Ramp", f"{sample}_Hold"
        if rk in processed_dfs and hk in processed_dfs:
            w_r = processed_dfs[rk].iloc[:, 0].values.astype(float)
            w_h = processed_dfs[hk].iloc[:, 0].values.astype(float)
            m_d = (w_r >= DIAG_MIN) & (w_r <= DIAG_MAX)
            if len(w_r) == len(w_h) and np.allclose(w_r, w_h, atol=1e-2):
                y_r_end = processed_dfs[rk].iloc[:, -1].values.astype(float)[m_d]
                y_h_t0 = processed_dfs[hk].iloc[:, 1].values.astype(float)[m_d]
                rms_cont = float(np.sqrt(np.mean((y_h_t0 - y_r_end) ** 2)))
                manifest_df.loc[manifest_df["Dataset"].isin([rk, hk]), "Continuity_RMS_1475_1750"] = round(rms_cont, 6)
                print(
                    f"  {sample:20s}: {rk}[{processed_dfs[rk].columns[-1]}] -> {hk}[{processed_dfs[hk].columns[1]}] "
                    f"| RMS_diff(1475-1750) = {rms_cont:.6f} Abs"
                )

    pd.DataFrame(audit_rows).to_csv(
        os.path.join(args.out_dir, "Step1A_Audit_Baseline_Offsets.csv"), index=False, encoding="utf-8-sig"
    )
    manifest_df.to_csv(
        os.path.join(args.out_dir, "Step1A_Manifest.csv"), index=False, encoding="utf-8-sig"
    )
    print(f"\n[DONE] Step 1A outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
