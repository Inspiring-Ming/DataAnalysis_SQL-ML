# Deploying the dashboard

The app is hosted on **Hugging Face Spaces** (Streamlit SDK). A public Space
opens directly — no viewer login — so visitors land straight on the app's
password prompt. The app is gated by `check_password()` in `app.py`, which
reads `st.secrets["app_password"]`; if that secret is unset the app runs open.

> ⚠️ A **public** Space (or public Git repo) makes its committed data files
> publicly downloadable. The `app_password` gate protects the *app UI*, not the
> raw files. Only publish data you have the right to redistribute; otherwise
> deploy from a **private** repo/Space, or ship an anonymized sample.

## What the Space contains

- `app.py`, `src/`, `requirements.txt`
- `README.md` with a Spaces YAML header (`sdk: streamlit`, `app_file: app.py`)
- `outputs/*.parquet` and `outputs/esg.db` (the app reads these at runtime;
  the ~960 MB `esg_full.db` is **not** shipped — exceeds size limits)

## Deploy / update (Hugging Face Spaces)

1. Create a Space at https://huggingface.co/new-space — SDK **Streamlit**,
   visibility **Public** (for a no-login link).
2. Push the app files and data (`app.py`, `src/`, `requirements.txt`,
   `outputs/`, and a `README.md` with the Spaces YAML header). The Space
   rebuilds automatically on every push.
3. In **Settings → Variables and secrets**, add a secret:
   `app_password = ming` (or your chosen password). It is kept out of the
   public file listing.
4. Pin the Streamlit version: set `sdk_version` in the README header **and**
   pin `streamlit==<same version>` in `requirements.txt`. A mismatch is a
   common cause of a Space that builds but never reaches **Running**.

Share the **direct app URL** — `https://<user>-<space>.hf.space/` — not the
`huggingface.co/spaces/...` wrapper page, so visitors land on the app (and the
password box) rather than the Files/Logs tabs.

## Alternative: Streamlit Community Cloud

Works the same way **but** apps deployed from a *private* GitHub repo force a
Streamlit login on every viewer — so the no-login + password-only flow needs a
*public* repo there. Steps: push to GitHub, deploy at https://share.streamlit.io
(repo, branch `main`, file `app.py`), then set `app_password` under
**Settings → Secrets**.

## Test the password gate locally

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # set a password
streamlit run app.py
```

`.streamlit/secrets.toml` is gitignored, so the real password is never
committed. With no secret set, the app runs open.
