# ESG Landscape Explorer

An end-to-end data-science project on a real, messy ESG dataset: **6.6M raw
observations → a clean company × metric matrix → PCA, clustering, industry
profiling, and a data-quality (disclosure-gap) analysis — served as an
interactive Streamlit dashboard with an in-app SQL console.**

Built on Clarity AI corporate ESG data (Feb 2025): ~65k companies, 96 metrics
across the Environmental / Social / Governance pillars, matched to industries
via perm-id.

> **Live demo (password-gated):** deployed on Hugging Face Spaces. The app is
> gated behind a single password read from `st.secrets["app_password"]`; if no
> password is configured it runs open (handy for local dev). See
> [DEPLOY.md](DEPLOY.md) for how it is hosted.

---

## Headline findings

- **PC1 ≈ "ESG disclosure maturity / environmental footprint."** The first
  principal component (17.5% of variance) loads jointly on energy use, CO₂,
  waste, water and emissions/labour policies. It separates large, heavily-
  reporting, environmentally-material firms from light disclosers.
- **Clusters split by disclosure intensity, with partial industry signal.**
  KMeans archetypes group heavy-industry firms (Chemicals, Industrial
  Machinery) into high-footprint clusters and asset-light firms (Software,
  Asset Management) into low-footprint ones — but industry alone does not
  determine the cluster.
- **84% of Environmental observations are *estimated*, vs ~20% for Social &
  Governance.** Companies disclose S/G data directly, but their environmental
  numbers are overwhelmingly modelled by the data provider — a material
  data-quality caveat for anyone using ESG scores.

---

## Pipeline

```
raw long-format CSVs  (6.6M rows, mixed units, 12 orders of magnitude)
        │  src/prepare_data.py
        ▼
  long_clean.parquet              tidy, numeric, latest-per-company-metric
  wide_matrix.parquet             57,771 companies × 95 metrics (33% filled)
  company_meta / metric_catalog   industry join (99.6% matched), coverage
  disclosure_long.parquet         reported-vs-estimated observation counts
        │  src/analysis.py
        ▼
  pca_scores / pca_loadings / pca_explained
  clusters.parquet                KMeans archetypes
  industry_profile.parquet        z-scored E/S/G profile per industry
        │  src/build_sqlite.py  (optional)
        ▼
  esg.db                          analytical tables as a queryable SQLite DB
        │  app.py
        ▼
  Streamlit dashboard (6 interactive views, incl. SQL console)
```

### Key data-science decisions (the actual work)

- **Long → wide pivot** with coverage thresholds (drop metrics seen for <2,000
  companies, companies with <5 metrics) so PCA is meaningful on sparse data.
- **Signed-log transform on heavy-tailed metrics.** Raw values span ~12 orders
  of magnitude (water withdrawal up to 9×10¹¹) with skew >200; without a log
  step a single mega-emitter dominates every component and the covariance
  matrix overflows float64. Binary Yes/No flags and bounded ratios are left
  untouched.
- **Median imputation + standardisation + ±8σ clip** before PCA.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/prepare_data.py            # build matrix + metadata  (~1-2 min)
python src/analysis.py                # PCA, clusters, profiles  (~30 s)
python -m src.build_sqlite --slim     # (optional) build esg.db for the SQL page
streamlit run app.py                  # launch dashboard
```

Or with the Makefile: `make all`, then `make sqlite` (optional), then `make app`.

Makefile targets: `make data` · `make analysis` · `make sqlite` (slim
`esg.db`, used by the app's SQL page) · `make sqlite-full` (local-only ~960 MB
DB with the raw observation tables) · `make app` · `make clean`.

---

## Dashboard views

| View | What it shows |
|------|---------------|
| **Overview** | Dataset scale, matrix fill, metric coverage by pillar |
| **PCA Explorer** | 2D projection (colour by industry/cluster), scree plot, per-component loadings |
| **Clusters** | KMeans archetypes in PCA space + industry composition (tune k) |
| **Industry Profiles** | z-scored E/S/G heatmap; rank industries on any metric |
| **Disclosure Gap** | Reported vs estimated share by pillar and by metric |
| **SQL Query** | Read-only `SELECT`/`WITH` console over `esg.db`, with example queries and CSV export |

---

## Layout

```
esg_project/
├── app.py                 Streamlit dashboard (incl. password gate + SQL page)
├── src/
│   ├── prepare_data.py    long → wide pipeline
│   ├── analysis.py        PCA / KMeans / profiling (importable + CLI)
│   └── build_sqlite.py    parquet → SQLite (slim esg.db / full esg_full.db)
├── outputs/               cached parquet artifacts + esg.db
├── .streamlit/
│   └── secrets.toml.example   app_password template (real secrets gitignored)
├── Makefile
├── requirements.txt
├── DEPLOY.md              hosting guide (Hugging Face Spaces)
└── README.md
```
