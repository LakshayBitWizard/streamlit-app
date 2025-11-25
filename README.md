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

Helper script: setup_repo_and_push.ps1
-----------------------------------
You can run the included helper PowerShell script `setup_repo_and_push.ps1` to initialize a git repo, optionally create a GitHub repository (if you have GitHub CLI installed) and push the code.

Examples:

1) If you already created a repo via the GitHub website:

```powershell
$env:GIT_REMOTE = 'https://github.com/<YOUR_USERNAME>/<REPO-NAME>.git'
./setup_repo_and_push.ps1
```

2) If you have `gh` (GitHub CLI) installed and want the script to create the repo for you:

```powershell
gh auth login
./setup_repo_and_push.ps1 -RepoName my-repo-name -Public
```

Remember: Git must be installed in your environment and you must be authenticated to GitHub to push.
# streamlit-app
