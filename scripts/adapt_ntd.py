#!/usr/bin/env python3
"""Adapt the Nano-Tumor Database (NTD, Chen et al. 2023 ACS Nano) for BIOPINN.

Usage:
    python scripts/adapt_ntd.py --input path/to/ntd_combined_long.csv   # or .xlsx
    python scripts/adapt_ntd.py --input ... --fdm-demo                   # + k_el FDM demo

The input is the NTD 2023 data in tidy/long form, one row per
(dataset, tissue, time point), with the columns listed in REQUIRED_COLUMNS
(an .xlsx needs a "Combined_Long_Format" sheet). See docs/NTD_ADAPTATION.md
for where the data comes from. The raw file is NOT committed; the small
derived tables it produces in data/external/ntd/ are.

BIOPINN trains on *synthetic* FDM fields
C(r,t) inside a 100-500 um avascular spheroid.  The NTD is *in vivo mouse
biodistribution*: whole-organ %ID vs time.  It therefore cannot be used as
L_data labels.  What it CAN legitimately supply:

  1. Plasma/blood pharmacokinetics -> an empirical elimination rate k_el
     (1/hr) per nanoparticle, which plugs straight into the time-varying
     Dirichlet boundary already implemented in src/fdm_solver.py
     (C(R,t) = C0 * exp(-k_el * t), argument `k_el_per_hr`).
  2. Literature-grounded parameter ranges (d_NP, k_el, t_max) for the LHS.
  3. External, qualitative validation targets: tumour uptake-vs-time shape
     and size trends that BIOPINN's volume-averaged tumour concentration can
     be compared against (direction / rank correlation only).

Outputs (in --out, default data/external/ntd/):
  ntd_clean.csv                 every row, parsed + QC flags (large; git-ignored)
  ntd_pk_fits.csv               per-dataset blood/plasma k_el fits
  ntd_tumor_delivery.csv        per-dataset tumour Cmax/Tmax/AUC/%ID@24/48/72h
  ntd_size_bins_summary.csv     validation targets binned like BIOPINN's d_NP range
  ntd_biopinn_scenarios.csv     BIOPINN-ready (d_NP_nm, k_el_per_hr, t_max_hr) rows
  k_el_config_suggestion.yaml   data-derived config values (already applied in configs/)
  ntd_summary.json              headline stats + correlations
  fdm_kel_demo.csv              (--fdm-demo) FDM demo: k_el=0 vs NTD median / p95
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "external" / "ntd"
REQUIRED_COLUMNS = (
    "Dataset_No", "Tissue", "Time_h", "Concentration_pctID", "Nanoparticle_ID", "Particle_Type",
    "Core_Material", "Targeting_Strategy", "Shape", "Size_raw", "Zeta_Potential_mV",
    "Polydispersity_Index", "Tumor_Cell", "Tumor_Size", "Dose_mg_per_kg",
)

# BIOPINN's own design space (configs/default_config.yaml -> dataset.parameter_ranges)
D_NP_RANGE = (10.0, 200.0)
T_MAX_RANGE = (24.0, 72.0)
PK_TISSUES = ["Blood", "Plasma", "Serum"]
NUM = r"(\d+(?:\.\d+)?)"
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 renamed trapz


# ---------------------------------------------------------------- parsers
def parse_size(raw: str) -> tuple[float, float]:
    """'TEM: 20; HD: 22.26' -> (TEM, HD) in nm (NaN if absent / 'NA')."""
    raw = str(raw)
    def grab(tag):
        m = re.search(tag + r"\s*:\s*" + NUM, raw)
        return float(m.group(1)) if m else np.nan
    return grab("TEM"), grab("HD")


def parse_tumor_volume_mm3(raw: str) -> tuple[float, str]:
    """Best-effort tumour volume in mm^3 + a note on how it was derived."""
    s = str(raw).replace(" ", "").replace("~", "").lower()
    # explicit cm3 value (usually after '='), take the last one, ranges -> midpoint
    m = re.findall(NUM + r"(?:-" + NUM + r")?cm3", s)
    if m:
        a, b = m[-1]
        v = (float(a) + float(b)) / 2 if b else float(a)
        return v * 1000.0, "cm3"
    m = re.search(NUM + r"(?:-" + NUM + r")?(?:±\d+(?:\.\d+)?)?mm3", s)
    if m:
        a, b = m.group(1), m.group(2)
        return ((float(a) + float(b)) / 2 if b else float(a)), "mm3"
    m = re.fullmatch(NUM + r"(?:-" + NUM + r")?mm", s)  # a diameter, e.g. '6-8 mm'
    if m:
        a, b = m.group(1), m.group(2)
        d = (float(a) + float(b)) / 2 if b else float(a)
        return 4 / 3 * np.pi * (d / 2) ** 3, "diameter_mm->sphere"
    m = re.fullmatch(NUM, s)  # bare number: <10 assumed cm3, else mm3
    if m:
        v = float(m.group(1))
        return (v * 1000.0, "bare_assumed_cm3") if v < 10 else (v, "bare_assumed_mm3")
    return np.nan, "unparsed"


def parse_dose(raw: str) -> tuple[str, float, float]:
    """'IV; 5 (20g BW)' -> ('IV', 5.0 mg/kg, 20.0 g). Radioactivity doses -> NaN."""
    s = str(raw)
    route = s.split(";")[0].strip() if ";" in s else s.strip()
    body = s.split(";", 1)[1] if ";" in s else ""
    dose = np.nan
    if body and not re.search(r"ci|bq", body, re.I):
        m = re.match(r"\s*" + NUM, body)
        dose = float(m.group(1)) if m else np.nan
    bw = np.nan
    m = re.search(NUM + r"(?:-" + NUM + r")?(?:±\d+)?\s*g\s*BW", s)
    if m:
        bw = (float(m.group(1)) + float(m.group(2))) / 2 if m.group(2) else float(m.group(1))
    return route, dose, bw


# ---------------------------------------------------------------- analysis
def fit_monoexp(t: np.ndarray, c: np.ndarray) -> dict:
    """ln C = a - k t (least squares on positive points)."""
    ok = (c > 0) & np.isfinite(c) & (t >= 0)
    t, c = t[ok], c[ok]
    if len(np.unique(t)) < 3:
        return {"k_el_per_hr": np.nan, "r2": np.nan, "n": int(len(t))}
    res = stats.linregress(t, np.log(c))
    return {"k_el_per_hr": -res.slope, "C_t0_pctID": float(np.exp(res.intercept)),
            "r2": res.rvalue ** 2, "n": int(len(t))}


def tumor_metrics(g: pd.DataFrame) -> dict:
    g = g.groupby("Time_h", as_index=False)["Concentration_pctID"].mean().sort_values("Time_h")
    t, c = g.Time_h.to_numpy(), g.Concentration_pctID.to_numpy()
    t0, c0 = np.r_[0.0, t], np.r_[0.0, c]  # tumour %ID is 0 at t=0
    out = {"n_tumor_pts": len(t), "t_last_h": t[-1], "Cmax_pctID": c.max(),
           "Tmax_h": t[c.argmax()], "C_last_pctID": c[-1],
           "AUC_0_last_pctID_h": _trapz(c0, t0)}
    for tp in (24, 48, 72):
        out[f"pctID_{tp}h"] = float(np.interp(tp, t, c)) if t[0] <= tp <= t[-1] else np.nan
    w = t0 <= 72
    out["AUC_0_72_pctID_h"] = _trapz(c0[w], t0[w]) if t[-1] >= 72 else np.nan
    return out


def load_ntd(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, sheet_name="Combined_Long_Format")
        except ImportError as exc:  # openpyxl isn't in requirements.txt
            raise SystemExit(f"Reading .xlsx needs openpyxl (pip install openpyxl), or pass the CSV instead: {exc}")
    else:
        df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing required column(s): {', '.join(missing)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, type=Path, help="NTD 2023 combined long-format CSV or XLSX")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--fdm-demo", action="store_true", help="also run the repo FDM solver with k_el=0 vs NTD median/p95")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_ntd(args.input)
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        df[col] = df[col].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()

    sz = df.Size_raw.map(parse_size)
    df["size_TEM_nm"] = [a for a, _ in sz]
    df["size_HD_nm"] = [b for _, b in sz]
    df["d_NP_nm"] = df.size_HD_nm.fillna(df.size_TEM_nm)          # Stokes-Einstein wants hydrodynamic
    df["d_NP_basis"] = np.where(df.size_HD_nm.notna(), "HD", np.where(df.size_TEM_nm.notna(), "TEM", "none"))
    tv = df.Tumor_Size.map(parse_tumor_volume_mm3)
    df["tumor_volume_mm3"] = [a for a, _ in tv]
    df["tumor_volume_note"] = [b for _, b in tv]
    df["tumor_R_equiv_um"] = (3 * df.tumor_volume_mm3 / (4 * np.pi)) ** (1 / 3) * 1000.0
    ds = df.Dose_mg_per_kg.map(parse_dose)
    df["route"] = [a for a, _, _ in ds]
    df["dose_mg_per_kg_num"] = [b for _, b, _ in ds]
    df["body_weight_g"] = [c for _, _, c in ds]
    df["tumor_pctID_per_g"] = np.where(df.Tissue == "Tumor",
                                       df.Concentration_pctID / (df.tumor_volume_mm3 / 1000.0), np.nan)  # ~1 g/cm3

    df["qc_gt100_pctID"] = df.Concentration_pctID > 100          # impossible for whole-organ %ID
    df["qc_zero"] = df.Concentration_pctID <= 0
    df["in_biopinn_scope"] = ((df.route == "IV") & (df.Shape == "Spherical")
                              & df.d_NP_nm.between(*D_NP_RANGE))
    df.to_csv(OUT / "ntd_clean.csv", index=False)

    meta_cols = ["Dataset_No", "Nanoparticle_ID", "Particle_Type", "Core_Material", "Targeting_Strategy",
                 "Shape", "d_NP_nm", "d_NP_basis", "Zeta_Potential_mV", "Polydispersity_Index",
                 "Tumor_Cell", "tumor_volume_mm3", "tumor_R_equiv_um", "dose_mg_per_kg_num", "route",
                 "in_biopinn_scope"]
    meta = df.groupby("Dataset_No")[meta_cols[1:]].first().reset_index()
    clean = df[~df.qc_gt100_pctID]

    # --- PK fits (one PK tissue per dataset: the one with most time points)
    pk_rows = []
    for dsn, g in clean[clean.Tissue.isin(PK_TISSUES)].groupby("Dataset_No"):
        tis = g.groupby("Tissue").Time_h.nunique().sort_values(ascending=False).index[0]
        gg = g[g.Tissue == tis].groupby("Time_h").Concentration_pctID.mean()
        t, c = gg.index.to_numpy(float), gg.to_numpy(float)
        full = fit_monoexp(t, c)
        win = fit_monoexp(t[t <= 72], c[t <= 72])  # BIOPINN's t_max window
        pk_rows.append({"Dataset_No": dsn, "pk_tissue": tis, "t_last_h": t.max(),
                        "k_el_per_hr_all": full["k_el_per_hr"], "r2_all": full["r2"],
                        "k_el_per_hr_0_72h": win["k_el_per_hr"], "r2_0_72h": win["r2"], "n_0_72h": win["n"]})
    pk = meta.merge(pd.DataFrame(pk_rows), on="Dataset_No")
    pk["t_half_h_0_72h"] = np.log(2) / pk.k_el_per_hr_0_72h
    pk["pk_fit_ok"] = (pk.k_el_per_hr_0_72h > 0) & (pk.r2_0_72h >= 0.8) & (pk.n_0_72h >= 3)
    pk.to_csv(OUT / "ntd_pk_fits.csv", index=False)

    # --- tumour delivery
    tum = pd.DataFrame([{"Dataset_No": dsn, **tumor_metrics(g)}
                        for dsn, g in clean[clean.Tissue == "Tumor"].groupby("Dataset_No")])
    tum = meta.merge(tum, on="Dataset_No")
    tum["Cnorm_24h"] = tum.pctID_24h / tum.Cmax_pctID  # shape metric comparable to C_avg(t)/C0
    tum.to_csv(OUT / "ntd_tumor_delivery.csv", index=False)

    # --- size-binned validation targets (BIOPINN scope only)
    bins = [10, 50, 100, 150, 200]
    both = tum.merge(pk[["Dataset_No", "k_el_per_hr_0_72h", "pk_fit_ok"]], on="Dataset_No", how="left")
    sc = both[both.in_biopinn_scope].copy()
    sc["d_NP_bin"] = pd.cut(sc.d_NP_nm, bins, include_lowest=True)
    agg = sc.groupby("d_NP_bin", observed=False).agg(
        n_datasets=("Dataset_No", "count"),
        tumor_Cmax_pctID_median=("Cmax_pctID", "median"),
        tumor_pctID_24h_median=("pctID_24h", "median"),
        tumor_Tmax_h_median=("Tmax_h", "median"),
        tumor_AUC_0_last_median=("AUC_0_last_pctID_h", "median"),
        n_pk=("pk_fit_ok", lambda s: int(s.fillna(False).sum())),
    ).reset_index()
    kel_bin = sc[sc.pk_fit_ok == True].groupby("d_NP_bin", observed=False).k_el_per_hr_0_72h.median()
    agg["k_el_per_hr_median"] = agg.d_NP_bin.map(kel_bin).astype(float)
    agg["d_NP_bin"] = agg.d_NP_bin.astype(str)
    agg.to_csv(OUT / "ntd_size_bins_summary.csv", index=False)

    # --- BIOPINN-ready scenarios: one row per in-scope dataset with a good PK fit
    good = pk[pk.pk_fit_ok & pk.in_biopinn_scope].copy()
    scen = pd.DataFrame({
        "ntd_dataset_no": good.Dataset_No,
        "nanoparticle_id": good.Nanoparticle_ID,
        "d_NP_nm": good.d_NP_nm.round(2),
        "d_NP_basis": good.d_NP_basis,
        "k_el_per_hr": good.k_el_per_hr_0_72h.round(5),
        "t_max_hr": good.t_last_h.clip(*T_MAX_RANGE),
        "pk_r2": good.r2_0_72h.round(3),
        "invivo_tumor_R_equiv_um": good.tumor_R_equiv_um.round(0),  # info only; NOT a spheroid radius
    })
    scen.to_csv(OUT / "ntd_biopinn_scenarios.csv", index=False)

    kel = good.k_el_per_hr_0_72h
    p5, p50, p95 = np.percentile(kel, [5, 50, 95]) if len(kel) else (np.nan,) * 3
    rho_kel = stats.spearmanr(good.d_NP_nm, kel) if len(kel) > 5 else None
    t24 = sc.dropna(subset=["pctID_24h"])
    rho_tum = stats.spearmanr(t24.d_NP_nm, t24.pctID_24h) if len(t24) > 5 else None

    yaml_txt = f"""# Data-derived values generated by scripts/adapt_ntd.py (Nano-Tumor Database 2023).
# These are what configs/default_config.yaml and configs/experiment_ntd.yaml use
# (rounded); regenerate and compare if the NTD input or the fit filters change.
dataset:
  parameter_ranges:
    d_NP_nm: [10, 200]                  # NTD in-scope HD diameters: median {good.d_NP_nm.median():.0f} nm (n={len(good)})
    k_el_per_hr: [{p5:.4f}, {p95:.4f}]   # NTD blood/plasma mono-exp fits over 0-72 h, 5th-95th pct (median {p50:.4f})
model:
  input_dim: 8                           # r, t, R, d_NP, C0, k_d, t_max, k_el
paper:
  baseline:
    k_el_per_hr: {p50:.4f}               # NTD median
"""
    (OUT / "k_el_config_suggestion.yaml").write_text(yaml_txt, encoding="utf-8")

    summary = {
        "source_file": args.input.name,
        "rows": int(len(df)), "datasets": int(df.Dataset_No.nunique()),
        "rows_gt100_pctID_excluded": int(df.qc_gt100_pctID.sum()),
        "rows_zero_pctID": int(df.qc_zero.sum()),
        "datasets_in_biopinn_scope(IV, spherical, 10-200nm)": int(meta.in_biopinn_scope.sum()),
        "datasets_with_PK_tissue": int(len(pk)),
        "datasets_good_PK_fit_in_scope": int(len(good)),
        "k_el_per_hr_p5_p50_p95": [p5, p50, p95],
        "t_half_h_median": float(np.log(2) / p50) if p50 == p50 else None,
        "spearman_dNP_vs_kel": None if rho_kel is None else [rho_kel.statistic, rho_kel.pvalue],
        "spearman_dNP_vs_tumor_pctID_24h": None if rho_tum is None else [rho_tum.statistic, rho_tum.pvalue, int(len(t24))],
        "tumor_R_equiv_um_quantiles_5_50_95": np.nanpercentile(meta.tumor_R_equiv_um, [5, 50, 95]).tolist(),
        "tumor_volume_parse_notes": meta.merge(df.groupby("Dataset_No").tumor_volume_note.first(), on="Dataset_No")
                                        .tumor_volume_note.value_counts().to_dict(),
        "d_NP_basis_counts_in_scope": meta[meta.in_biopinn_scope].d_NP_basis.value_counts().to_dict(),
        "tumor_Cmax_pctID_median_in_scope": float(sc.Cmax_pctID.median()),
        "tumor_Tmax_h_median_in_scope": float(sc.Tmax_h.median()),
    }
    (OUT / "ntd_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print(agg.to_string())

    # --- optional: demonstrate the existing k_el hook in the repo's FDM solver
    if args.fdm_demo and p50 == p50:
        sys.path.insert(0, str(REPO))
        from src.config import load_config
        from src.fdm_solver import solve_fdm
        cfg = load_config()
        cfg["fdm"]["N_t_initial"] = 145  # coarse output grid is fine for a demo
        rows = []
        for kel_v in (0.0, float(p50), float(p95)):
            res = solve_fdm(400.0, 100.0, 10.0, 0.01, 72.0, cfg, k_el_per_hr=kel_v)
            r, C = res["r"], res["C"]
            w = r ** 2
            for ti in (24, 48, 72):
                idx = int(np.argmin(np.abs(res["t"] - ti)))
                Ci = C[idx]
                pen = r[-1] - r[np.argmax(Ci >= 1.0)] if (Ci >= 1.0).any() else 0.0  # C >= 1 uM (= IC50)
                rows.append({"k_el_per_hr": kel_v, "t_hr": ti, "C_surface_uM": Ci[-1],
                             "C_volavg_uM": float((Ci * w).sum() / w.sum()),
                             "penetration_depth_um_(C>=1uM)": float(pen)})
        demo = pd.DataFrame(rows)
        demo.to_csv(OUT / "fdm_kel_demo.csv", index=False)
        print(demo.to_string())


if __name__ == "__main__":
    main()
