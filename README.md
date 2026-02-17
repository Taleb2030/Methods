# Methods — Constrained Optimization

Optimization methods (penalty, barrier, Zoutendijk, etc.) with a web interface.

## Run locally (Visual Studio)

1. Open `Method_SHtof.slnx` in **Visual Studio 2022** (or 2026).
2. Press **F5** or click **Run**.
3. The app starts a server at **http://127.0.0.1:8000/** and opens the page in your browser. Use the forms to run each method.

## See it live on the web (GitHub Pages)

After you push this repo to GitHub, you can serve the same UI from GitHub so anyone can open it in a browser **without** running Visual Studio.

1. Push your code to GitHub (e.g. `https://github.com/Taleb2030/Methods`).
2. On GitHub: open the repo → **Settings** → **Pages**.
3. Under **Build and deployment**:
   - **Source**: Deploy from a branch
   - **Branch**: `master` (or `main`)
   - **Folder**: `/docs`
4. Click **Save**. After a minute or two, the site will be at:
   - **https://taleb2030.github.io/Methods/**

On that page, **Penalty** and **Barrier** methods run in the browser. The other methods still need the C++ server (run the project in Visual Studio and use http://localhost:8000).

## Repo structure

- `Method_SHtof/` — C++ project (server + all 7 methods)
- `docs/` — Static web app for GitHub Pages (methods 1–2 run in browser)
