# ESG Landscape Explorer

An end-to-end data-science project on a real, messy ESG dataset: **6.6M raw
observations → a clean company × metric matrix → PCA, clustering, industry
profiling, and a data-quality (disclosure-gap) analysis — served as an
interactive Streamlit dashboard.**

Built on Clarity AI corporate ESG data (Feb 2025): ~65k companies, 96 metrics
across the Environmental / Social / Governance pillars, matched to industries
via perm-id.

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
        │  app.py
        ▼
  Streamlit dashboard (5 interactive views)
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

python src/prepare_data.py     # build matrix + metadata  (~1-2 min)
python src/analysis.py         # PCA, clusters, profiles  (~30 s)
streamlit run app.py           # launch dashboard
```

Or with the Makefile: `make all` then `make app`.

---

## Dashboard views

| View | What it shows |
|------|---------------|
| **Overview** | Dataset scale, matrix fill, metric coverage by pillar |
| **PCA Explorer** | 2D projection (colour by industry/cluster), scree plot, per-component loadings |
| **Clusters** | KMeans archetypes in PCA space + industry composition (tune k) |
| **Industry Profiles** | z-scored E/S/G heatmap; rank industries on any metric |
| **Disclosure Gap** | Reported vs estimated share by pillar and by metric |

---

## Layout

```
esg_project/
├── app.py                 Streamlit dashboard
├── src/
│   ├── prepare_data.py    long → wide pipeline
│   └── analysis.py        PCA / KMeans / profiling (importable + CLI)
├── outputs/               cached parquet artifacts
├── requirements.txt
└── README.md
```
