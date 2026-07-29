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


# shared loaders for the carbon workstream pages (WS1-WS5)
@st.cache_data(show_spinner=False)
def _carbon_audit():
    return (pd.read_parquet(f"{OUT}/carbon_summary.parquet"),
            pd.read_parquet(f"{OUT}/carbon_per_metric.parquet"),
            pd.read_parquet(f"{OUT}/carbon_audit_log.parquet"))


@st.cache_data(show_spinner=False)
def _carbon_by_year():
    return pd.read_parquet(f"{OUT}/carbon_panel_by_year.parquet")


# --------------------------------------------------------------------------- #
st.sidebar.title("ESG Landscape Explorer")

# Two clearly separated sections: general ESG exploration, and the carbon
# research workstreams (July analysis plan). A prefix keeps the flat radio
# visually grouped; PAGES maps the label back to a plain page key.
PAGES = {
    "Overview": "Overview",
    "ESG — Main patterns (PCA)": "PCA Explorer",
    "ESG — Company groups": "Clusters",
    "ESG — Industry profiles": "Industry Profiles",
    "Carbon 1 — Understanding the data": "Carbon Data Audit",
    "Carbon 2 — Sourcing over time": "Carbon Panel",
    "Carbon 3 — Grouping the metrics": "Carbon Taxonomy",
    "Carbon 4 — Comparison samples": "Carbon Samples",
    "Carbon 5 — Reported vs estimated (PCA)": "Carbon PCA",
    "SQL query tool": "SQL Query",
}
choice = st.sidebar.radio("View", list(PAGES), label_visibility="collapsed")
page = PAGES[choice]
st.sidebar.caption(
    "**ESG** — general exploration of all 95 sustainability metrics.\n\n"
    "**Carbon** — a five-step look at the carbon data, following the research "
    "plan."
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
    st.title("ESG — Main patterns across the metrics (PCA)")
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
    st.title("ESG — Groups of similar companies")
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
    st.title("ESG — Industry profiles")
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


# ===================== CARBON · WS1 DATA AUDIT ============================= #
elif page == "Carbon Data Audit":
    st.title("Step 1 — Understanding the carbon data")
    st.markdown(
        "This is the starting point for everything that follows. Before running "
        "any analysis, we look at exactly what the carbon data contains: how many "
        "companies and years it covers, how each metric is measured and sourced, "
        "and how much of it is missing. Every later step (the trends, the "
        "samples, the PCA) draws on the same data described here."
    )

    summ, pm, log = _carbon_audit()
    s = summ.iloc[0]

    st.subheader("The dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{int(s['carbon_observations']):,}")
    c2.metric("Companies", f"{int(s['companies']):,}")
    c3.metric("Carbon metrics", int(s["carbon_metrics"]))
    c4.metric("Years covered", f"{int(s['year_min'])}–{int(s['year_max'])}")

    st.subheader("Each metric in detail")
    st.markdown(
        "One row per carbon metric, showing how many companies have it, how many "
        "years it spans, how it is sourced, and the size of the numbers. The "
        "three source columns — **reported**, **estimated**, **calculated** — "
        "add up to 100% for each metric. This is the ground truth the rest of "
        "the analysis relies on."
    )
    view = pm[pm["n"] >= 100].copy()
    # source shares from the actual counts, so all three always sum to 100%
    view["rep"] = (view["REPORTED"] / view["n"] * 100).round(0).astype(int)
    view["est"] = (view["ESTIMATED"] / view["n"] * 100).round(0).astype(int)
    view["calc"] = (view["CALCULATED"] / view["n"] * 100).round(0).astype(int)
    tbl = view.assign(**{
        "metric": view["metric_name"],
        "type": view["group"],
        "unit": view["unit"],
        "companies": view["companies"],
        "coverage": (view["coverage"] * 100).round(0).astype(int).astype(str) + "%",
        "years": view["first_year"].astype(int).astype(str) + "–"
                 + view["last_year"].astype(int).astype(str),
        "reported %": view["rep"],
        "estimated %": view["est"],
        "calculated %": view["calc"],
        "median value": view["value_median"].round(1),
    })
    st.dataframe(
        tbl[["metric", "type", "unit", "companies", "coverage", "years",
             "reported %", "estimated %", "calculated %", "median value"]]
        .sort_values("companies", ascending=False),
        use_container_width=True, hide_index=True)
    st.caption(
        "How to read the source columns: emission metrics (Scope 1/2/3, NOx, "
        "SOx) are mostly **estimated** by the provider; policy and target "
        "metrics are **reported** or **calculated**, never estimated. Emission "
        "metrics cover almost every company, while renewable-energy and target "
        "metrics cover only a small fraction. (A fourth state, 'adjusted', "
        "appears on just 2 observations and is negligible.)"
    )

    st.subheader("How much data is missing")
    st.markdown(
        "The values that are recorded are clean — no blank or impossible "
        "numbers. The real limitation is coverage: most companies simply do not "
        "have most metrics in most years."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Company–metric–year gaps",
              f"{s['structural_missing_share']*100:.0f}%",
              help="Out of every possible company × metric × year combination, "
                   "this share does not exist in the data.")
    m2.metric("Latest snapshot empty",
              f"{s['matrix_missing_share']*100:.0f}%",
              help="Using each company's most recent value per metric, this "
                   "share of the company × metric grid is still blank.")
    m3.metric("No observation date",
              f"{s['reported_date_missing_share']*100:.0f}%",
              help="Share of observations with no recorded reporting date.")

    st.subheader("Quality checks")
    st.caption("Each check, what it looks at, and the result. 'PASS' means the "
               "recorded values are sound; 'LIMITATION' flags the missing-data "
               "gaps above — real caveats, not errors in the values.")
    st.dataframe(log, use_container_width=True, hide_index=True)


# ================== CARBON · WS2 PANEL & TRENDS =========================== #
elif page == "Carbon Panel":
    st.title("Step 2 — How the carbon data is sourced, over time")
    st.markdown(
        "Each carbon value comes from one of three sources: the company "
        "**reported** it, the provider **estimated** it, or it was "
        "**calculated**. Using the 2016–2024 data, this page shows how that mix "
        "has changed, and which metrics companies are increasingly reporting "
        "themselves."
    )

    by_year = _carbon_by_year()

    st.subheader("Where the data comes from, each year")
    st.caption("Every carbon value per year, split by source. More green over "
               "time means companies are reporting their own numbers instead of "
               "the provider estimating them.")
    mix = by_year[by_year["year"] <= 2023].copy()
    mix["reported"] = mix["reported_share"] * mix["observations"]
    mix["estimated"] = mix["estimated_share"] * mix["observations"]
    mix["other"] = mix["observations"] - mix["reported"] - mix["estimated"]
    agg = mix.groupby("year")[["reported", "estimated", "other"]].sum().reset_index()
    m2 = agg.melt(id_vars="year", var_name="source", value_name="obs")
    fig = px.area(m2, x="year", y="obs", color="source",
                  labels={"obs": "observations", "year": ""},
                  color_discrete_map={"reported": "#2e8b57",
                                      "estimated": "#d9822b", "other": "#4682b4"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Share reported, by metric")
    st.caption("For each metric, the share of companies reporting it themselves. "
               "A rising line means more companies report that metric over time.")
    metrics_ts = sorted(by_year["metric_name"].unique())
    default_ts = [m for m in ["CO2DIRECTSCOPE1", "CO2INDIRECTSCOPE2",
                              "ENERGYUSETOTAL"] if m in metrics_ts]
    pick = st.multiselect("Metrics", metrics_ts,
                          default=default_ts or metrics_ts[:3])
    ts = by_year[(by_year["metric_name"].isin(pick)) & (by_year["year"] <= 2023)]
    if not ts.empty:
        fig2 = px.line(ts, x="year", y="reported_share", color="metric_name",
                       markers=True, labels={"reported_share": "% reported",
                                             "year": "", "metric_name": "metric"})
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("2024 excluded as a partial vintage.")
    else:
        st.info("Select at least one metric.")


# ==================== CARBON · WS3 TAXONOMY =============================== #
elif page == "Carbon Taxonomy":
    st.title("Step 3 — Grouping the carbon metrics")
    st.markdown(
        "The carbon metrics measure different things, so we don't lump them into "
        "one score. Each metric is labelled two ways:\n\n"
        "- **What it measures** — *exposure* (actual emissions and energy use) or "
        "*commitment* (policies and targets the company sets).\n"
        "- **Where the emissions come from** — *direct* (the company's own "
        "operations: Scope 1 and air pollutants), *indirect* (Scope 2 and 3, "
        "from purchased energy and the supply chain), or *energy* use."
    )

    _, pm, _ = _carbon_audit()
    pm = pm[pm["n"] >= 100].copy()

    st.subheader("How often each emission type is company-reported")
    st.caption("Companies report their direct emissions the least and their "
               "energy use the most — so data reliability depends on the "
               "emission type.")
    dim = pm[pm["group"] == "exposure"].copy()
    byloc = dim.groupby("emission").apply(
        lambda x: pd.Series({
            "reported_share": (x["reported_share"] * x["n"]).sum() / x["n"].sum(),
            "observations": x["n"].sum()}), include_groups=False).reset_index()
    fig = px.bar(byloc, x="emission", y="reported_share", color="emission",
                 text=(byloc["reported_share"] * 100).round(0).astype(int)
                 .astype(str) + "%",
                 labels={"reported_share": "% reported", "emission": ""})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metric dictionary")
    tbl = pm.assign(**{
        "reported %": (pm["reported_share"] * 100).round(1),
        "estimated %": (pm["estimated_share"] * 100).round(1),
    }).rename(columns={"metric_name": "metric", "n": "observations"})
    st.dataframe(
        tbl[["metric", "group", "emission", "unit", "observations",
             "companies", "reported %", "estimated %"]].sort_values(
                 ["group", "emission", "observations"],
                 ascending=[True, True, False]),
        use_container_width=True, hide_index=True)


# ==================== CARBON · WS4 SAMPLES =============================== #
elif page == "Carbon Samples":
    st.title("Step 4 — Four groups of companies to compare")
    st.markdown(
        "A number a company reported and a number the provider estimated are not "
        "the same kind of data, so we don't just mix them together. Instead we "
        "look at four groups:\n\n"
        "- **All** — every company we have a value for.\n"
        "- **Reported** — only companies that reported the number themselves.\n"
        "- **Estimated** — only companies whose number was estimated by the "
        "provider.\n"
        "- **Fair comparison** — a balanced group, explained at the bottom of "
        "the page."
    )

    SAMPLE_LABEL = {"all": "All", "reported": "Reported",
                    "estimated": "Estimated", "common_support": "Fair comparison"}
    comp = pd.read_parquet(f"{OUT}/carbon_sample_compare.parquet")
    show = comp.assign(group=comp["sample"].map(SAMPLE_LABEL)).rename(columns={
        "matrix_fill": "matrix fill", "scope1_median_tons": "Scope-1 median (t)"})
    st.subheader("The four groups side by side")
    st.dataframe(show[["group", "firms", "metrics", "matrix fill",
                       "Scope-1 median (t)"]],
                 use_container_width=True, hide_index=True)
    st.caption(
        "Companies that report their own Scope-1 emissions are far bigger "
        "emitters — a median of about 8,200 tonnes, versus about 440 tonnes for "
        "companies whose figure is estimated (roughly 19× higher). So the "
        "provider is mostly estimating the smaller companies that don't report."
    )

    comp2 = comp.assign(group=comp["sample"].map(SAMPLE_LABEL))
    fig = px.bar(comp2, x="group", y="firms", color="group", text="firms",
                 labels={"firms": "companies", "group": ""})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why we build a \"fair comparison\" group")
    st.markdown(
        "There's a catch in comparing reported vs. estimated companies directly: "
        "the reported ones are big emitters and the estimated ones are small. So "
        "if the two groups look different, we can't tell whether it's because of "
        "**how the number was produced** or simply because **big and small "
        "companies are different**.\n\n"
        "To remove the size difference, we pair up companies of similar size — "
        "each reporting company is matched with an estimated company that emits "
        "about the same amount (like matching people of the same height before "
        "comparing their weight). Keeping only these matched pairs gives the "
        "**fair-comparison group**: reported and estimated companies that are "
        "alike in size.\n\n"
        "Now any remaining difference between them is down to reporting vs. "
        "estimation — not size. This is the group we trust most when comparing "
        "the two data sources."
    )


# ===================== CARBON · WS5 PCA DIAGNOSTIC ======================== #
elif page == "Carbon PCA":
    st.title("Step 5 — Do reported and estimated data tell the same story?")
    st.markdown(
        "**The question (from the plan).** Is there a single, meaningful "
        "\"carbon\" pattern in the data — and does it hold up whether the numbers "
        "were reported by companies or estimated by the provider? If it holds up, "
        "the project can focus on carbon *exposure*. If it falls apart once we "
        "separate reported from estimated, the story is really about data "
        "*quality*."
    )
    with st.expander("What is PCA doing here, in plain terms?"):
        st.markdown(
            "Each company has several carbon numbers (Scope 1, 2, 3, energy, air "
            "pollutants). These tend to move together — a heavy emitter is high "
            "on most of them. **PCA finds the one combined score that best "
            "captures that shared movement** — think of it as an overall "
            "\"carbon-intensity\" score built from all the metrics at once.\n\n"
            "Two things tell us if that score is meaningful:\n"
            "- **How much it explains (PC1 %)** — a high number means one carbon "
            "score really does summarise most of the differences between "
            "companies.\n"
            "- **What goes into it (loadings)** — the weight each metric carries. "
            "If the weights look the same across the four samples, the carbon "
            "score means the same thing regardless of how the data was sourced."
        )

    @st.cache_data(show_spinner=False)
    def carbon_pca_tables():
        return (pd.read_parquet(f"{OUT}/carbon_pca_explained.parquet"),
                pd.read_parquet(f"{OUT}/carbon_pca_loadings.parquet"))

    cexp, cload = carbon_pca_tables()
    SAMPLE_NAME = {"all": "All", "reported": "Reported",
                   "estimated": "Estimated", "common_support": "Fair comparison"}

    st.subheader("Is there one strong carbon score? (higher = yes)")
    st.caption("How much of the differences between companies the single carbon "
               "score captures, in each sample.")
    pc1 = cexp[cexp["PC"] == "PC1"].copy()
    pc1["sample_name"] = pc1["sample"].map(SAMPLE_NAME)
    fig = px.bar(pc1, x="sample_name", y="explained", color="sample_name",
                 text=pc1["explained"].mul(100).round(0).astype(int).astype(str) + "%",
                 labels={"explained": "share explained by the carbon score",
                         "sample_name": ""},
                 hover_data=["n_firms", "n_metrics"])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The carbon score captures a large share in every sample (35–63%), so a "
        "single carbon-intensity measure is meaningful whether the data is "
        "reported or estimated."
    )

    st.subheader("What goes into the carbon score? (should match across samples)")
    st.caption(
        "Each metric's weight in the score, for every sample. The weights line up "
        "on Scope 1/2/3, energy and air pollutants across all samples — so the "
        "score means the same thing regardless of data source. (Renewables "
        "appear only in the reported sample, because the provider does not "
        "estimate renewable-energy use.)"
    )
    cl = cload.copy()
    cl["sample_name"] = cl["sample"].map(SAMPLE_NAME)
    fig3 = px.bar(cl, x="PC1", y="metric_name", color="sample_name",
                  orientation="h", barmode="group", height=650,
                  labels={"PC1": "weight in the carbon score", "metric_name": "",
                          "sample_name": "sample"})
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("What this means for the research direction")
    st.markdown(
        "The carbon score is **strong and consistent** across all four samples — "
        "reported, estimated, and the matched common-support set all produce the "
        "same emissions-based factor. Separating reported from estimated data "
        "does **not** break it.\n\n"
        "This points to the plan's **first option: a carbon-exposure result** — "
        "the project can focus on the pricing and materiality of real carbon "
        "exposure, rather than treating estimation quality as the main story. "
        "(The provider's estimates broadly reproduce the same structure as "
        "company-reported figures.)"
    )

    with st.expander("Method details"):
        st.markdown(
            "- **Data.** One row per company, its most recent value for each "
            "carbon-exposure metric, from the 2016–2024 panel (metrics kept if "
            "held by ≥500 companies; companies kept with ≥3 metrics).\n"
            "- **Processing.** Log-scale the large emission values, fill gaps "
            "with the median, standardise, cap extreme outliers, then run PCA — "
            "the same steps as the main ESG PCA, carbon metrics only.\n"
            "- **Note.** The broad 95-metric ESG PCA mixes emissions with policy "
            "and governance; restricting to carbon metrics gives a cleaner "
            "carbon-only factor, as the plan asked."
        )

    with st.expander("WS6 — what stays in the research pipeline"):
        st.markdown(
            "This platform covers the diagnostic steps (WS1–WS5). **WS6 — "
            "linking carbon data to stock returns and financing costs for the "
            "pricing tests — stays in the private research pipeline**, since "
            "return-linked financial data should not sit behind a shared "
            "password. Only about **12% of observations carry a reported date**, "
            "so building realistic information-timing rules will need care and is "
            "best done in that pipeline."
        )


# ============================== SQL QUERY ================================== #
elif page == "SQL Query":
    import sqlite3

    st.title("SQL query tool")
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
