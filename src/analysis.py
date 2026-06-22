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

    print("\nDONE. Analysis artifacts cached to outputs/")


if __name__ == "__main__":
    main()
