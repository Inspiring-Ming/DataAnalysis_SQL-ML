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
# Carbon workstreams (July analysis plan). The carbon universe is grouped by
# economic meaning (WS3 taxonomy): EXPOSURE = physical emissions/energy outcomes;
# COMMITMENT = policies/targets; the rest of ESG is out of the carbon scope.
CARBON_METRICS = {
    "CO2DIRECTSCOPE1": "exposure",
    "CO2INDIRECTSCOPE2": "exposure",
    "CO2INDIRECTSCOPE3": "exposure",
    "CO2_NO_EQUIVALENTS": "exposure",
    "ENERGYUSETOTAL": "exposure",
    "NOXEMISSIONS": "exposure",
    "SOXEMISSIONS": "exposure",
    "VOCEMISSIONS": "exposure",
    "PARTICULATE_MATTER_EMISSIONS": "exposure",
    "RENEWENERGYCONSUMED": "exposure",
    "RENEWENERGYPRODUCED": "exposure",
    "RENEWENERGYPURCHASED": "exposure",
    "POLICY_EMISSIONS": "commitment",
    "TARGETS_EMISSIONS": "commitment",
}


def carbon_audit(meta: pd.DataFrame) -> dict:
    """WS1 + WS3: reproducible audit of the carbon subset of long_clean.

    Returns aggregate frames (no raw rows) suitable for the app:
      * per_metric      -- provenance split, unit(s), coverage, group.
      * summary         -- one-row dataset-level facts (rows, firms, year state).
    Provenance is kept as the full 3-way REPORTED / ESTIMATED / CALCULATED.
    """
    lc = pd.read_parquet(os.path.join(OUT, "long_clean.parquet"))
    cb = lc[lc["metric_name"].isin(CARBON_METRICS)].copy()
    cb["group"] = cb["metric_name"].map(CARBON_METRICS)

    prov = (cb.groupby(["metric_name", "disclosure"]).size()
              .unstack(fill_value=0))
    for state in ["REPORTED", "ESTIMATED", "CALCULATED"]:
        if state not in prov.columns:
            prov[state] = 0
    prov = prov[["REPORTED", "ESTIMATED", "CALCULATED"]]
    prov["n"] = prov.sum(axis=1)
    prov["reported_share"] = prov["REPORTED"] / prov["n"]
    prov["estimated_share"] = prov["ESTIMATED"] / prov["n"]

    per_metric = prov.reset_index()
    per_metric["group"] = per_metric["metric_name"].map(CARBON_METRICS)
    per_metric["unit"] = cb.groupby("metric_name")["metric_unit"].first().values
    per_metric["companies"] = cb.groupby("metric_name")["perm_id"].nunique().values

    summary = pd.DataFrame([{
        "carbon_observations": len(cb),
        "companies": cb["perm_id"].nunique(),
        "carbon_metrics": cb["metric_name"].nunique(),
        "null_identifiers": int(cb["perm_id"].isna().sum()),
        "duplicate_firm_metric": int(cb.duplicated(["perm_id", "metric_name"]).sum()),
        "null_values": int(cb["metric_value"].isna().sum()),
        "metrics_multi_unit": int((cb.groupby("metric_name")["metric_unit"]
                                   .nunique() > 1).sum()),
        "has_year_dimension": bool(cb["metric_year"].notna().any()),
    }])
    return {"carbon_per_metric": per_metric, "carbon_summary": summary}


def carbon_pca_diagnostic() -> dict:
    """WS4 + WS5: run carbon-only PCA on parallel provenance samples.

    Builds a firm x carbon-EXPOSURE-metric matrix (latest year per firm-metric)
    for each regime (all / reported-only / estimated-only), then runs the same
    impute-scale-PCA on each. Returns, per regime, the explained-variance of the
    first components and PC1 loadings, so the app can show whether the carbon
    factor structure is stable when reported and estimated data are separated.
    """
    panel = pd.read_parquet(os.path.join(OUT, "carbon_panel.parquet"))
    exp = panel[panel["group"] == "exposure"].copy()
    exp = exp.sort_values("year").drop_duplicates(
        ["perm_id", "metric_name"], keep="last")

    regimes = {
        "all": exp,
        "reported": exp[exp["disclosure"] == "REPORTED"],
        "estimated": exp[exp["disclosure"] == "ESTIMATED"],
    }
    explained_rows, loading_frames = [], []
    for name, sub in regimes.items():
        wide = sub.pivot_table(index="perm_id", columns="metric_name",
                               values="value_num", aggfunc="last")
        # keep metrics present for >=500 firms; firms with >=3 metrics
        wide = wide.loc[:, wide.notna().sum() >= 500]
        wide = wide[wide.notna().sum(axis=1) >= 3]
        if wide.shape[1] < 3 or len(wide) < 100:
            continue
        Xs, _, _ = impute_scale(wide)
        n_comp = min(5, Xs.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
        pca.fit(Xs)
        for i, ev in enumerate(pca.explained_variance_ratio_):
            explained_rows.append({"regime": name, "PC": f"PC{i+1}",
                                   "explained": ev, "n_firms": len(wide),
                                   "n_metrics": wide.shape[1]})
        ld = pd.DataFrame({"regime": name, "metric_name": wide.columns,
                           "PC1": pca.components_[0]})
        loading_frames.append(ld)

    return {
        "carbon_pca_explained": pd.DataFrame(explained_rows),
        "carbon_pca_loadings": pd.concat(loading_frames, ignore_index=True),
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

    print("building disclosure breakdowns ...")
    for name, frame in disclosure_breakdowns(meta).items():
        frame.to_parquet(os.path.join(OUT, f"{name}.parquet"))
        print(f"  {name}: {len(frame):,} rows")

    print("building carbon audit (WS1) ...")
    for name, frame in carbon_audit(meta).items():
        frame.to_parquet(os.path.join(OUT, f"{name}.parquet"))
        print(f"  {name}: {len(frame):,} rows")

    print("\nDONE. Analysis artifacts cached to outputs/")


if __name__ == "__main__":
    main()
