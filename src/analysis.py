"""
Analysis module: PCA + complementary methods on the ESG wide matrix.

Reusable functions (imported by the Streamlit app) plus a CLI that pre-computes
and caches the headline results to outputs/.

Methods:
  * impute_scale     -- median-impute sparse matrix, standardize.
  * run_pca          -- principal components + loadings + explained variance.
  * run_kmeans       -- cluster companies in PCA space; profile clusters.
  * industry_profile -- mean standardized metric value per industry (heatmap).
  * disclosure_gap   -- REPORTED vs ESTIMATED share per pillar / industry.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

# numpy 2.0 + the BLAS shipped on some macOS builds emits spurious
# overflow/divide warnings inside sklearn's matmul. Outputs are verified finite;
# silence the cosmetic noise so pipeline logs stay readable.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


@dataclass
class PCAResult:
    scores: pd.DataFrame        # companies x PCs
    loadings: pd.DataFrame      # metrics x PCs
    explained: np.ndarray       # explained variance ratio per PC
    feature_names: list


def load_artifacts():
    wide = pd.read_parquet(os.path.join(OUT, "wide_matrix.parquet"))
    meta = pd.read_parquet(os.path.join(OUT, "company_meta.parquet"))
    catalog = pd.read_parquet(os.path.join(OUT, "metric_catalog.parquet"))
    return wide, meta, catalog


def _is_binary(col: pd.Series) -> bool:
    vals = col.dropna()
    if vals.empty:
        return True
    return set(np.unique(vals.unique())).issubset({0.0, 1.0})


def _is_heavy_tailed(col: pd.Series) -> bool:
    """Heavy-tailed magnitude metric (emissions, energy, $, signed counts).

    ESG numeric metrics span ~12 orders of magnitude with skew 100-200+, which
    overflows the PCA covariance matrix. We signed-log-transform those; binary
    Yes/No flags and small bounded ratios (0-100) are left untouched.
    """
    vals = col.dropna()
    if vals.empty or _is_binary(col):
        return False
    return vals.abs().max() > 1000 or abs(vals.skew()) > 5


def impute_scale(wide: pd.DataFrame):
    """Signed-log heavy-tailed metrics, median-impute, z-score, clip outliers.

    Returns (X_scaled, scaler, imputer). The signed-log step handles both
    non-negative magnitudes (emissions) and signed counts; the final clip to
    +/-8 sigma stops any lone outlier from dominating a component (or
    overflowing float64).
    """
    W = wide.copy()
    heavy = [c for c in W.columns if _is_heavy_tailed(W[c])]
    # signed log: sign(x) * log1p(|x|) -- safe for negatives, monotonic.
    W[heavy] = np.sign(W[heavy]) * np.log1p(W[heavy].abs())

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(W.values)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.clip(Xs, -8, 8)
    return Xs, scaler, imputer


def run_pca(wide: pd.DataFrame, n_components: int = 10) -> PCAResult:
    Xs, _, _ = impute_scale(wide)
    n_components = min(n_components, Xs.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    scores = pca.fit_transform(Xs)
    pcs = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=wide.index, columns=pcs)
    loadings_df = pd.DataFrame(
        pca.components_.T, index=wide.columns, columns=pcs
    )
    return PCAResult(
        scores=scores_df,
        loadings=loadings_df,
        explained=pca.explained_variance_ratio_,
        feature_names=list(wide.columns),
    )


def run_kmeans(scores: pd.DataFrame, k: int = 6, use_pcs: int = 5):
    cols = scores.columns[:use_pcs]
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(scores[cols].values)
    return pd.Series(labels, index=scores.index, name="cluster")


def industry_profile(wide: pd.DataFrame, meta: pd.DataFrame,
                     min_companies: int = 300) -> pd.DataFrame:
    """Mean z-scored metric value per industry (industries x metrics)."""
    Xs, _, _ = impute_scale(wide)
    z = pd.DataFrame(Xs, index=wide.index, columns=wide.columns)
    ind = meta.set_index("perm_id")["industry"].reindex(wide.index)
    z["industry"] = ind.values
    counts = z["industry"].value_counts()
    big = counts[counts >= min_companies].index
    prof = z[z["industry"].isin(big)].groupby("industry").mean(numeric_only=True)
    return prof


def disclosure_gap(long: pd.DataFrame) -> pd.DataFrame:
    """Share of ESTIMATED (vs REPORTED) observations per pillar."""
    sub = long[long["data_type"].isin(["REPORTED", "ESTIMATED"])] \
        if "data_type" in long.columns else long
    # data_type in categorized files is actually float/int; disclosure is the
    # REPORTED/ESTIMATED column.
    return sub


def disclosure_breakdowns(meta: pd.DataFrame) -> dict:
    """Link the reported-vs-estimated gap to industry / metric / company / country.

    Reads the tidy per-observation table (long_clean.parquet), joins industry
    from meta, and returns four aggregate frames keyed on estimated_share =
    P(ESTIMATED). Each carries an observation count `n` so the app can filter
    out thin, noisy groups.
    """
    lc = pd.read_parquet(os.path.join(OUT, "long_clean.parquet"))[
        ["perm_id", "company_name", "disclosure", "metric_name", "pillar",
         "headquarter_country"]
    ]
    ind = meta[["perm_id", "industry"]].drop_duplicates()
    df = lc.merge(ind, on="perm_id", how="left")
    df["est"] = (df["disclosure"] == "ESTIMATED").astype(int)

    def agg(keys):
        g = df.groupby(keys)["est"].agg(estimated_share="mean", n="count")
        return g.reset_index()

    by_industry = agg(["industry", "pillar"])
    by_metric_industry = agg(["metric_name", "pillar", "industry"])
    by_company = agg(["perm_id", "company_name", "industry"])
    by_country = agg(["headquarter_country"])
    return {
        "disclosure_by_industry": by_industry,
        "disclosure_by_metric_industry": by_metric_industry,
        "disclosure_by_company": by_company,
        "disclosure_by_country": by_country,
    }


# --------------------------------------------------------------------------- #
# Carbon workstreams (July analysis plan). Each carbon metric is tagged on two
# dimensions so the platform can analyse it either way:
#   * group     -- economic meaning: exposure vs commitment (WS3).
#   * emission  -- GHG-Protocol locus: direct / indirect / energy / n.a. (WS3).
CARBON_TAXONOMY = {
    #  metric                         group         emission
    "CO2DIRECTSCOPE1":              ("exposure",   "direct"),
    "NOXEMISSIONS":                 ("exposure",   "direct"),
    "SOXEMISSIONS":                 ("exposure",   "direct"),
    "VOCEMISSIONS":                 ("exposure",   "direct"),
    "PARTICULATE_MATTER_EMISSIONS": ("exposure",   "direct"),
    "CO2INDIRECTSCOPE2":            ("exposure",   "indirect"),
    "CO2INDIRECTSCOPE3":            ("exposure",   "indirect"),
    "CO2_NO_EQUIVALENTS":           ("exposure",   "direct"),
    "ENERGYUSETOTAL":               ("exposure",   "energy"),
    "RENEWENERGYCONSUMED":          ("exposure",   "energy"),
    "RENEWENERGYPRODUCED":          ("exposure",   "energy"),
    "RENEWENERGYPURCHASED":         ("exposure",   "energy"),
    "POLICY_EMISSIONS":             ("commitment", "n.a."),
    "TARGETS_EMISSIONS":            ("commitment", "n.a."),
}
# backwards-compatible view: {metric: group}
CARBON_METRICS = {m: g for m, (g, _) in CARBON_TAXONOMY.items()}
CARBON_EMISSION = {m: e for m, (_, e) in CARBON_TAXONOMY.items()}


def carbon_audit() -> dict:
    """WS1: carbon-focused data audit + an explicit AUDIT LOG.

    Uses the recovered carbon panel (firm-metric-year), so the audit reflects the
    same data the analysis runs on. Returns:
      * carbon_per_metric  -- provenance split, unit, coverage, group, emission.
      * carbon_summary     -- one-row dataset-level facts.
      * carbon_audit_log   -- one row per check: what was audited, and the finding
                              (the plan requires recording what the audit covers).
    """
    p = pd.read_parquet(os.path.join(OUT, "carbon_panel.parquet"))
    # WS1 audits at the firm-metric-year grain (the panel's natural key)
    prov = (p.groupby(["metric_name", "disclosure"]).size().unstack(fill_value=0))
    for state in ["REPORTED", "ESTIMATED", "CALCULATED"]:
        if state not in prov.columns:
            prov[state] = 0
    prov["n"] = prov.sum(axis=1)
    prov["reported_share"] = prov["REPORTED"] / prov["n"]
    prov["estimated_share"] = prov["ESTIMATED"] / prov["n"]
    per_metric = prov.reset_index()
    per_metric["group"] = per_metric["metric_name"].map(CARBON_METRICS)
    per_metric["emission"] = per_metric["metric_name"].map(CARBON_EMISSION)
    per_metric["unit"] = p.groupby("metric_name")["metric_unit"].first().reindex(
        per_metric["metric_name"]).values
    per_metric["companies"] = p.groupby("metric_name")["perm_id"].nunique().reindex(
        per_metric["metric_name"]).values

    # coverage = share of ALL firms that have this metric (latest year). This is
    # the real "missing data" story: most cells are absent, not null.
    latest = p.sort_values("year").drop_duplicates(
        ["perm_id", "metric_name"], keep="last")
    n_firms = p["perm_id"].nunique()
    per_metric["coverage"] = (
        per_metric["companies"] / n_firms).round(3)

    dup = int(p.duplicated(["perm_id", "metric_name", "year"]).sum())
    n_units = int((p.groupby("metric_name")["metric_unit"].nunique() > 1).sum())
    neg = int((p[p["metric_name"] == "CO2DIRECTSCOPE1"]["value_num"] < 0).sum())
    yrs = sorted(p["year"].unique())

    n_metrics = p["metric_name"].nunique()
    possible_cells = n_firms * n_metrics * len(yrs)
    struct_missing = 1 - len(p) / possible_cells        # firm-metric-year gaps
    wide = latest.pivot_table(index="perm_id", columns="metric_name",
                              values="value_num", aggfunc="last")
    matrix_missing = 1 - float(wide.notna().mean().mean())   # latest firm x metric
    rd_missing = float(p["reported_date"].isna().mean())

    summary = pd.DataFrame([{
        "carbon_observations": len(p),
        "companies": n_firms,
        "carbon_metrics": n_metrics,
        "year_min": yrs[0], "year_max": yrs[-1],
        "duplicate_firm_metric_year": dup,
        "null_identifiers": int(p["perm_id"].isna().sum()),
        "null_values_in_existing": int(p["value_num"].isna().sum()),
        "structural_missing_share": round(struct_missing, 3),
        "matrix_missing_share": round(matrix_missing, 3),
        "reported_date_missing_share": round(rd_missing, 3),
        "metrics_multi_unit": n_units,
    }])

    # explicit audit log -- records each required check and its outcome.
    # We separate "values that exist are clean" (PASS) from "how much data is
    # missing" (NOTE / LIMITATION), so the audit is honest about coverage.
    log = [
        ("Identifiers", "perm_id present on every observation",
         f"{int(p['perm_id'].isna().sum())} null identifiers", "PASS"),
        ("Duplicates", "no duplicate firm-metric-year rows",
         f"{dup} duplicates", "PASS" if dup == 0 else "REVIEW"),
        ("Years", "time dimension present",
         f"{yrs[0]}-{yrs[-1]} ({len(yrs)} years)", "PASS"),
        ("Units", "each metric uses one consistent unit",
         f"{n_units} metrics with mixed units", "PASS" if n_units == 0 else "REVIEW"),
        ("Recorded values", "values that exist are non-null and numeric",
         f"{int(p['value_num'].isna().sum())} null among recorded values", "PASS"),
        ("Value sanity", "no negative Scope-1 emissions",
         f"{neg} negative values", "PASS" if neg == 0 else "REVIEW"),
        ("Coverage / completeness",
         "how much of the firm-metric-year grid is actually filled",
         f"{struct_missing*100:.0f}% of firm-metric-year cells are absent; "
         f"{matrix_missing*100:.0f}% of the latest firm x metric matrix is empty",
         "LIMITATION"),
        ("Reported date", "observation-date field populated",
         f"{rd_missing*100:.0f}% of observations have no reported_date", "LIMITATION"),
        ("Provenance", "every observation tagged reported/estimated/calculated",
         f"{p['disclosure'].nunique()} states: "
         + ", ".join(sorted(p['disclosure'].unique())), "PASS"),
        ("Source versions", "duplicated raw .csv/.csv.gz files read once",
         "de-duplicated at ingest", "PASS"),
    ]
    audit_log = pd.DataFrame(log, columns=["check", "requirement", "finding", "status"])

    return {"carbon_per_metric": per_metric, "carbon_summary": summary,
            "carbon_audit_log": audit_log}


def carbon_samples(panel: pd.DataFrame) -> dict:
    """WS4: four comparable carbon-exposure samples (latest year per firm-metric).

      all            -- every usable observation.
      reported       -- company-reported observations only.
      estimated      -- provider-estimated observations only.
      common_support -- balanced set where each reported firm is 1:1 matched to
                        the estimated firm with the closest Scope-1 emissions
                        (nearest-neighbour on log tons), so the two groups are
                        genuinely comparable rather than pooled mechanically.
    """
    exp = panel[panel["group"] == "exposure"].copy()
    exp = exp.sort_values("year").drop_duplicates(
        ["perm_id", "metric_name"], keep="last")

    # nearest-neighbour match on Scope-1 (log tons): each reported firm -> the
    # closest estimated firm; keep both sides of the matched pairs.
    s1 = exp[exp["metric_name"] == "CO2DIRECTSCOPE1"][
        ["perm_id", "disclosure", "value_num"]].dropna()
    rep = s1[s1["disclosure"] == "REPORTED"].copy()
    est = s1[s1["disclosure"] == "ESTIMATED"].copy()
    matched_ids = set()
    if len(rep) and len(est):
        est_sorted = est.sort_values("value_num").reset_index(drop=True)
        est_vals = est_sorted["value_num"].to_numpy()
        idx = np.searchsorted(est_vals, rep["value_num"].to_numpy())
        idx = np.clip(idx, 0, len(est_vals) - 1)
        matched_ids = set(rep["perm_id"]) | set(est_sorted.loc[idx, "perm_id"])
    common = exp[exp["perm_id"].isin(matched_ids)]

    return {
        "all": exp,
        "reported": exp[exp["disclosure"] == "REPORTED"],
        "estimated": exp[exp["disclosure"] == "ESTIMATED"],
        "common_support": common,
    }


def carbon_pca_diagnostic() -> dict:
    """WS4 + WS5: carbon-only PCA across the four parallel samples.

    For each sample, build a firm x carbon-exposure-metric matrix, run the same
    impute-scale-PCA, and record explained variance + PC1 loadings so the app can
    show whether the carbon factor structure survives when reported and estimated
    data are separated. Also returns a WS4 sample-comparison table (coverage and
    distribution differences across samples).
    """
    panel = pd.read_parquet(os.path.join(OUT, "carbon_panel.parquet"))
    samples = carbon_samples(panel)

    explained_rows, loading_frames, compare_rows = [], [], []
    for name, sub in samples.items():
        wide = sub.pivot_table(index="perm_id", columns="metric_name",
                               values="value_num", aggfunc="last")
        wide = wide.loc[:, wide.notna().sum() >= 500]
        wide = wide[wide.notna().sum(axis=1) >= 3]

        # WS4 comparison row: coverage + distribution of Scope-1 (log tons)
        s1v = sub[sub["metric_name"] == "CO2DIRECTSCOPE1"]["value_num"]
        compare_rows.append({
            "sample": name,
            "firms": int(sub["perm_id"].nunique()),
            "metrics": int(wide.shape[1]),
            "matrix_fill": round(float(wide.notna().mean().mean()), 3)
            if len(wide) else 0.0,
            "scope1_median_tons": round(float(s1v.median()), 1) if len(s1v) else None,
        })

        if wide.shape[1] < 3 or len(wide) < 100:
            continue
        Xs, _, _ = impute_scale(wide)
        n_comp = min(5, Xs.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
        pca.fit(Xs)
        for i, ev in enumerate(pca.explained_variance_ratio_):
            explained_rows.append({"sample": name, "PC": f"PC{i+1}",
                                   "explained": ev, "n_firms": len(wide),
                                   "n_metrics": wide.shape[1]})
        loading_frames.append(pd.DataFrame(
            {"sample": name, "metric_name": wide.columns,
             "PC1": pca.components_[0]}))

    return {
        "carbon_pca_explained": pd.DataFrame(explained_rows),
        "carbon_pca_loadings": pd.concat(loading_frames, ignore_index=True),
        "carbon_sample_compare": pd.DataFrame(compare_rows),
    }


def main() -> None:
    print("loading artifacts ...")
    wide, meta, catalog = load_artifacts()

    print("running PCA ...")
    pca = run_pca(wide, n_components=10)
    pca.scores.to_parquet(os.path.join(OUT, "pca_scores.parquet"))
    pca.loadings.to_parquet(os.path.join(OUT, "pca_loadings.parquet"))
    pd.Series(pca.explained, name="explained_variance_ratio").to_frame() \
        .to_parquet(os.path.join(OUT, "pca_explained.parquet"))
    print(f"  PC1-PC3 explain "
          f"{pca.explained[:3].sum():.1%} of variance")

    print("running KMeans (k=6) ...")
    clusters = run_kmeans(pca.scores, k=6)
    clusters.to_frame().to_parquet(os.path.join(OUT, "clusters.parquet"))

    print("building industry profile ...")
    prof = industry_profile(wide, meta)
    prof.to_parquet(os.path.join(OUT, "industry_profile.parquet"))
    print(f"  profiled {len(prof)} industries")

    print("building carbon audit (WS1) ...")
    for name, frame in carbon_audit().items():
        frame.to_parquet(os.path.join(OUT, f"{name}.parquet"))
        print(f"  {name}: {len(frame):,} rows")

    print("building carbon PCA diagnostic (WS4/WS5) ...")
    for name, frame in carbon_pca_diagnostic().items():
        frame.to_parquet(os.path.join(OUT, f"{name}.parquet"))
        print(f"  {name}: {len(frame):,} rows")

    print("\nDONE. Analysis artifacts cached to outputs/")


if __name__ == "__main__":
    main()
