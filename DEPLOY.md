# Deploying to Streamlit Community Cloud (password-protected)

The app is already prepared: it ships its data (`outputs/*.parquet`) and is gated
behind a password read from `st.secrets["app_password"]`.

> ⚠️ **Use a PRIVATE GitHub repo.** The Clarity AI ESG data and ~58k company
> names are likely licensed. A password protects the *app*, not the *repo* — a
> public repo would expose the raw data. Streamlit Cloud deploys from private
> repos for free.

## 1. Push to a private GitHub repo

From `esg_project/` (already a git repo with one commit on `main`):

```bash
gh repo create esg-landscape-explorer --private --source=. --remote=origin --push
```

(Or create the repo in the GitHub UI, then `git remote add origin <url> && git push -u origin main`.)

## 2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **New app** → pick the `esg-landscape-explorer` repo, branch `main`,
   main file `app.py`.
3. Click **Deploy**. First build installs `requirements.txt` (~2–3 min).

## 3. Set the password

In the app's **Settings → Secrets**, paste:

```toml
app_password = "your-strong-password-here"
```

Save. The app restarts and the password gate goes live. Share the URL
(`https://<name>.streamlit.app`) and the password with whoever needs access.

## Notes
- No password set → app is open (handy for local dev). The gate only activates
  once `app_password` exists in secrets.
- To test the gate locally: copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml`, set a password, then `streamlit run app.py`.
- Data lives in the repo (48 MB) and loads at runtime — no rebuild needed.
