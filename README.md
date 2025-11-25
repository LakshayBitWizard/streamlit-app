# Quick deploy steps for Streamlit Community Cloud

Follow these simple steps to connect this project to GitHub and deploy on Streamlit Community Cloud.

1. Initialize a local git repo and commit your code:

```powershell
git init
git add .
git commit -m "Initial commit"
```

2. Create a new GitHub repository (either via the website or the GitHub CLI `gh`).
   - Website: Go to https://github.com/new and create a repo.
   - OR, if you have GitHub CLI installed:

```powershell
gh repo create <USERNAME>/<REPO-NAME> --public --source=. --remote=origin --push
```

3. OR add the remote and push manually:

```powershell
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPO-NAME>.git
git push -u origin main
```

4. Deploy on Streamlit Community Cloud
   - Go to https://share.streamlit.io
   - Click *New app*, choose GitHub, select the repo and branch (usually `main`), and set `app.py` as the file path.

Notes:
- Keep `requirements.txt` in the repo root (already present). Streamlit Cloud will use it to install dependencies.
- If your model file is very large (>100 MB), use Git LFS or host it externally (S3, Google Drive) and download at runtime in `app.py`.
