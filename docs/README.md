# Project website (`docs/`)

A static, **zero-build** site (Tailwind via the Play CDN) detailing the journey, specs, and
benchmarks of the engine. One file: `docs/index.html`.

## View locally

Open `docs/index.html` directly in a browser, or serve the folder:

```bash
python -m http.server -d docs 8080      # -> http://localhost:8080
```

## Publish with GitHub Pages

Repo **Settings → Pages → Source: "Deploy from a branch"**, branch `main`, folder **`/docs`**.
The site goes live at `https://<user>.github.io/<repo>/`.

## Note

Repo/JOURNEY links in `index.html` use a `https://github.com/` placeholder — find-and-replace it
with your repository URL before publishing.
