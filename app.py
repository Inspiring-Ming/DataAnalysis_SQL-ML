"""
ESG Landscape Explorer -- Streamlit dashboard.

An interactive tour of ~58k companies x 95 ESG metrics:
  1. Overview      -- dataset scale, coverage, pillar balance.
  2. PCA Explorer  -- 2D projection coloured by industry/cluster, loadings.
  3. Clusters      -- KMeans archetypes and their industry composition.
  4. Industry      -- E/S/G profile heatmap and rankings.
  5. Disclosure    -- reported vs estimated gap (data-quality lens).

Run:  streamlit run app.py
Artifacts are pre-computed by src/prepare_data.py and src/analysis.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import run_kmeans

OUT = os.path.join(os.path.dirname(__file__), "outputs")

st.set_page_config(page_title="ESG Landscape Explorer", layout="wide")


# --------------------------------------------------------------------------- #
def check_password() -> bool:
    """Gate the app behind a password stored in st.secrets['app_password'].

    If no password is configured (e.g. local dev), the app is open. Set the
    secret in Streamlit Cloud → Settings → Secrets, or in .streamlit/secrets.toml.
    """
    expected = st.secrets.get("app_password", None)
    if not expected:  # no password configured -> open access (local dev)
        return True
    if st.session_state.get("authed"):
        return True

    st.title("ESG Landscape Explorer")
    with st.form("login", clear_on_submit=False):
        pw = st.text_input("Enter password to view", type="password")
        submitted = st.form_submit_button("Enter")
    if not submitted:
        st.stop()
    if pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    else:
        st.error("Incorrect password.")
        st.stop()
    return False


check_password()


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load():
    return {
        "wide": pd.read_parquet(f"{OUT}/wide_matrix.parquet"),
        "meta": pd.read_parquet(f"{OUT}/company_meta.parquet").set_index("perm_id"),
        "catalog": pd.read_parquet(f"{OUT}/metric_catalog.parquet"),
        "scores": pd.read_parquet(f"{OUT}/pca_scores.parquet"),
        "loadings": pd.read_parquet(f"{OUT}/pca_loadings.parquet"),
        "explained": pd.read_parquet(f"{OUT}/pca_explained.parquet"),
        "clusters": pd.read_parquet(f"{OUT}/clusters.parquet"),
        "profile": pd.read_parquet(f"{OUT}/industry_profile.parquet"),
    }


D = load()
PILLAR_NAME = {"E": "Environmental", "S": "Social", "G": "Governance"}
PILLAR_COLOR = {"E": "#2e8b57", "S": "#4682b4", "G": "#9370db"}


def scores_with_meta(top_n_industries: int | None = 12):
    df = D["scores"].join(D["meta"][["company_name", "industry"]])
    df = df.join(D["clusters"])
    df["cluster"] = df["cluster"].astype("Int64").astype(str)
    if top_n_industries:
        top = df["industry"].value_counts().head(top_n_industries).index
        df["industry_grp"] = np.where(df["industry"].isin(top),
                                      df["industry"], "Other")
    return df


# --------------------------------------------------------------------------- #
st.sidebar.title("ESG Landscape Explorer")
page = st.sidebar.radio(
    "View",
    ["Overview", "PCA Explorer", "Clusters", "Industry Profiles",
     "Disclosure Gap", "SQL Query"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(D['wide']):,} companies · {D['wide'].shape[1]} metrics · "
    "Source: Clarity AI ESG (Feb 2025)"
)


# ============================== OVERVIEW ==================================== #
if page == "Overview":
    st.title("ESG Landscape Explorer")
    st.markdown(
        "From **6.6M raw long-format observations** to an analysis-ready "
        "**company × metric matrix**, then PCA, clustering and industry "
        "profiling. Use the sidebar to explore each analysis."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(D['wide']):,}")
    c2.metric("Metrics", D["wide"].shape[1])
    c3.metric("Matrix fill", f"{D['wide'].notna().mean().mean():.0%}")
    known = (D["meta"].reindex(D["wide"].index)["industry"] != "Unknown").mean()
    c4.metric("Industry-matched", f"{known:.0%}")

    st.subheader("Metric coverage by pillar")
    cat = D["catalog"].copy()
    cat["pillar_name"] = cat["pillar"].map(PILLAR_NAME)
    fig = px.bar(
        cat.sort_values("n_companies", ascending=False).head(30),
        x="n_companies", y="metric_name", color="pillar_name", orientation="h",
        color_discrete_map={PILLAR_NAME[k]: v for k, v in PILLAR_COLOR.items()},
        labels={"n_companies": "Companies reporting", "metric_name": "",
                "pillar_name": "Pillar"},
        height=650,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ============================== PCA ======================================== #
elif page == "PCA Explorer":
    st.title("PCA Explorer")
    ev = D["explained"].values.ravel()
    st.markdown(
        f"**PC1** ({ev[0]:.0%} of variance) separates companies by ESG "
        "disclosure intensity & environmental footprint. Together the first "
        f"three PCs explain **{ev[:3].sum():.0%}**."
    )

    with st.expander("How is each PC's variance calculated? (click to read)"):
        st.markdown(
            "**1. A PC is a recipe over all 95 metrics.** Each metric gets a "
            "weight called a *loading*. A company's PC1 score is the weighted "
            "blend of its 95 (z-scored) metric values:\n\n"
            "`PC1 = w₁·metric₁ + w₂·metric₂ + … + w₉₅·metric₉₅`\n\n"
            "**2. Variance = how spread out those PC scores are.** Once every "
            "company has a single PC1 number, we take the ordinary statistical "
            "variance of that column. PCA chooses the weights that make this "
            "spread **as large as possible** — that's why PC1 is always the "
            "biggest slice.\n\n"
            "**3. The % you see** is that PC's variance ÷ the total variance "
            "across all components. PC1 here = "
            f"**{ev[0]:.1%}** of the total.\n\n"
            "**Every PC uses all 95 metrics — just with different weights.** "
            "Squaring a PC's loadings gives each metric's *share* of that "
            "component (they sum to 100%); the table under the bar chart below "
            "shows this."
        )

    df = scores_with_meta()
    colA, colB = st.columns([3, 1])
    with colB:
        color_by = st.radio("Colour by", ["Industry", "Cluster"])
        x_pc = st.selectbox("X axis", df.columns[:5].tolist(), index=0)
        y_pc = st.selectbox("Y axis", df.columns[:5].tolist(), index=1)
        sample = st.slider("Points shown", 2000, 20000, 8000, step=2000)

    plot_df = df.sample(min(sample, len(df)), random_state=0)
    color_col = "industry_grp" if color_by == "Industry" else "cluster"
    with colA:
        fig = px.scatter(
            plot_df, x=x_pc, y=y_pc, color=color_col,
            hover_data=["company_name", "industry"], opacity=0.6, height=600,
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scree plot — variance explained")
    scree = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(ev))],
        "explained": ev,
        "cumulative": np.cumsum(ev),
    })
    fig2 = px.bar(scree, x="PC", y="explained")
    fig2.add_scatter(x=scree["PC"], y=scree["cumulative"], mode="lines+markers",
                     name="cumulative")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("What drives each component? (top loadings)")
    pc = st.selectbox("Component", D["loadings"].columns.tolist())
    cat = D["catalog"].set_index("metric_name")
    load = D["loadings"][pc].sort_values()
    top = pd.concat([load.head(8), load.tail(8)]).to_frame("loading")
    top["pillar"] = cat["pillar"].reindex(top.index).map(PILLAR_NAME)
    fig3 = px.bar(
        top.reset_index(), x="loading", y="metric_name", color="pillar",
        orientation="h",
        color_discrete_map={PILLAR_NAME[k]: v for k, v in PILLAR_COLOR.items()},
        height=500,
    )
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        f"**Each metric's share of {pc}** — squared loadings, which sum to "
        "100% across all 95 metrics. The bar above shows *direction* "
        "(±); this shows *how much each metric drives the component*."
    )
    share = (D["loadings"][pc] ** 2).sort_values(ascending=False)
    share_df = share.head(15).rename("share").to_frame()
    share_df["share"] = (share_df["share"] * 100).round(2)
    share_df["pillar"] = cat["pillar"].reindex(share_df.index).map(PILLAR_NAME)
    share_df = share_df.reset_index().rename(
        columns={"metric_name": "metric", "share": "share of component (%)"}
    )
    st.dataframe(share_df, use_container_width=True, hide_index=True)


# ============================== CLUSTERS =================================== #
elif page == "Clusters":
    st.title("Company Archetypes (KMeans)")
    k = st.slider("Number of clusters (k)", 3, 10, 6)
    if k != 6:
        labels = run_kmeans(D["scores"], k=k)
    else:
        labels = D["clusters"]["cluster"]
    df = D["scores"].join(D["meta"][["industry"]])
    df["cluster"] = labels.astype(int)

    st.subheader("Clusters in PCA space")
    plot_df = df.sample(min(8000, len(df)), random_state=0)
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color=plot_df["cluster"].astype(str),
        opacity=0.6, height=550, labels={"color": "cluster"},
    )
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Industry composition of each cluster")
    comp = (
        df.groupby("cluster")["industry"].value_counts()
        .groupby(level=0).head(5).rename("companies").reset_index()
    )
    fig2 = px.bar(comp, x="companies", y="cluster", color="industry",
                  orientation="h", height=500)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Clusters split primarily by ESG disclosure intensity; industry "
        "signal is present but partial — heavy-industry firms concentrate in "
        "high-footprint clusters, asset-light firms in low-footprint ones."
    )


# ============================ INDUSTRY ===================================== #
elif page == "Industry Profiles":
    st.title("Industry ESG Profiles")
    st.markdown(
        "Mean **z-scored** metric value per industry (only industries with "
        "≥300 companies). Red = above average, blue = below."
    )
    with st.expander("What is a z-score? (click to read)"):
        st.markdown(
            "A **z-score** rescales every metric into the same unit: "
            "*how many standard deviations a value sits from the average*. "
            "The average and spread are computed across **all "
            f"{len(D['wide']):,} companies** in the dataset, so each metric is "
            "judged against the whole population.\n\n"
            "- **z = 0** → exactly average\n"
            "- **z = +1** → one standard deviation **above** average (notably high)\n"
            "- **z = −1** → one standard deviation **below** average (notably low)\n"
            "- **z = +2** → far above average (roughly top ~2.5%)\n\n"
            "In the heatmap below, each cell is an industry's **average z-score** "
            "for that metric. So a red cell at *(Oil & Gas, emissions)* means that "
            "industry emits well **above** the typical company — it does **not** "
            "mean the industry simply reports more data. Z-scoring lets metrics on "
            "wildly different scales (tonnes of CO₂ vs. a 0/1 policy flag) be "
            "compared fairly on one colour scale."
        )
    prof = D["profile"]
    cat = D["catalog"].set_index("metric_name")
    pillar_pick = st.multiselect(
        "Pillars", ["E", "S", "G"], default=["E", "S", "G"],
        format_func=lambda p: PILLAR_NAME[p],
    )
    cols = [c for c in prof.columns if cat["pillar"].get(c) in pillar_pick]
    show = prof[cols]
    fig = px.imshow(
        show, aspect="auto", color_continuous_scale="RdBu_r",
        zmin=-1.5, zmax=1.5, height=750,
        labels=dict(color="z-score"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rank industries on a single metric")
    metric = st.selectbox("Metric", sorted(prof.columns))
    rank = prof[metric].sort_values(ascending=False).reset_index()
    fig2 = px.bar(rank, x=metric, y="industry", orientation="h", height=600)
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)


# ============================ DISCLOSURE =================================== #
elif page == "Disclosure Gap":
    st.title("Disclosure Gap: Reported vs Estimated")
    st.markdown(
        "ESG data mixes company-**reported** figures with provider-"
        "**estimated** ones. The reliance on estimates is itself a "
        "data-quality signal."
    )

    @st.cache_data(show_spinner="Computing disclosure gap…")
    def disclosure():
        long = pd.read_parquet(f"{OUT}/disclosure_long.parquet")
        g = (
            long.groupby(["pillar", "disclosure"]).size()
            .unstack(fill_value=0)
        )
        g["est_share"] = g.get("ESTIMATED", 0) / g.sum(axis=1)
        return g, long

    g, long = disclosure()
    g = g.reset_index()
    g["pillar_name"] = g["pillar"].map(PILLAR_NAME)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            g, x="pillar_name", y="est_share",
            color="pillar_name",
            color_discrete_map={PILLAR_NAME[k]: v
                                for k, v in PILLAR_COLOR.items()},
            labels={"est_share": "Share estimated", "pillar_name": "Pillar"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        melt = g.melt(id_vars="pillar_name",
                      value_vars=["REPORTED", "ESTIMATED"],
                      var_name="type", value_name="n")
        fig2 = px.bar(melt, x="pillar_name", y="n", color="type",
                      barmode="group", labels={"n": "Observations",
                                               "pillar_name": "Pillar"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Most estimate-reliant metrics")
    mm = (
        long.groupby(["metric_name", "disclosure"]).size()
        .unstack(fill_value=0)
    )
    mm["est_share"] = mm.get("ESTIMATED", 0) / mm.sum(axis=1)
    mm["total"] = mm.get("ESTIMATED", 0) + mm.get("REPORTED", 0)
    top = mm[mm["total"] > 5000].sort_values("est_share", ascending=False).head(15)
    fig3 = px.bar(top.reset_index(), x="est_share", y="metric_name",
                  orientation="h", height=550,
                  labels={"est_share": "Share estimated", "metric_name": ""})
    fig3.update_xaxes(tickformat=".0%")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)


# ============================== SQL QUERY ================================== #
elif page == "SQL Query":
    import sqlite3

    st.title("SQL Query")
    DB_PATH = f"{OUT}/esg.db"
    if not os.path.exists(DB_PATH):
        st.error(
            "esg.db not found. Build it with: `python -m src.build_sqlite --slim`"
        )
        st.stop()

    st.markdown(
        "Run read-only **SQL** against the analysis tables. "
        "Only `SELECT`/`WITH` queries are allowed."
    )

    @st.cache_resource(show_spinner=False)
    def _conn():
        # read-only connection, shared across reruns
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                               check_same_thread=False)

    con = _conn()
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
    )["name"].tolist()

    with st.expander("Tables in this database"):
        for t in tables:
            cols = pd.read_sql(f'PRAGMA table_info("{t}")', con)["name"].tolist()
            n = pd.read_sql(f'SELECT COUNT(*) AS n FROM "{t}"', con)["n"][0]
            preview = ", ".join(cols[:8]) + (" …" if len(cols) > 8 else "")
            st.markdown(f"**{t}** ({n:,} rows) — {preview}")

    examples = {
        "Top 10 highest-emitting companies": (
            "SELECT m.company_name, m.industry, w.CO2DIRECTSCOPE1\n"
            "FROM wide_matrix w\n"
            "JOIN company_meta m ON m.perm_id = w.perm_id\n"
            "WHERE w.CO2DIRECTSCOPE1 IS NOT NULL\n"
            "ORDER BY w.CO2DIRECTSCOPE1 DESC\n"
            "LIMIT 10;"
        ),
        "Company count per industry": (
            "SELECT industry, COUNT(*) AS companies\n"
            "FROM company_meta\n"
            "GROUP BY industry\n"
            "ORDER BY companies DESC\n"
            "LIMIT 20;"
        ),
        "Cluster sizes": (
            "SELECT cluster, COUNT(*) AS companies\n"
            "FROM clusters\n"
            "GROUP BY cluster\n"
            "ORDER BY cluster;"
        ),
        "Strongest PC1 loadings": (
            "SELECT metric_name, PC1\n"
            "FROM pca_loadings\n"
            "ORDER BY ABS(PC1) DESC\n"
            "LIMIT 15;"
        ),
    }
    pick = st.selectbox("Example query (optional)", ["—"] + list(examples))
    default_sql = examples.get(pick, "SELECT * FROM company_meta LIMIT 20;")

    sql = st.text_area("SQL", value=default_sql, height=160)
    run = st.button("Run query", type="primary")

    if run:
        low = sql.strip().lower().rstrip(";")
        if not (low.startswith("select") or low.startswith("with")):
            st.error("Only SELECT / WITH queries are allowed.")
        elif any(f" {kw} " in f" {low} " for kw in
                 ("insert", "update", "delete", "drop", "alter", "create",
                  "attach", "pragma")):
            st.error("Write/DDL statements are not allowed.")
        else:
            try:
                # hard row cap so a huge result can't hang the app
                res = pd.read_sql(f"SELECT * FROM ({sql.rstrip(';')}) LIMIT 5000",
                                  con)
                st.success(f"{len(res):,} rows (capped at 5,000)")
                st.dataframe(res, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV", res.to_csv(index=False).encode(),
                    file_name="query_result.csv", mime="text/csv",
                )
            except Exception as e:  # noqa: BLE001 -- surface SQL errors to user
                st.error(f"Query error: {e}")
