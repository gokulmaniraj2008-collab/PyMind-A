# PyMind-A Vercel preview

This folder is a separate, lightweight Vercel project root. It contains a mobile-friendly static UI and Python serverless functions that use only the Python standard library.

## Vercel project settings

When importing `gokulmaniraj2008-collab/PyMind-A` into Vercel, set **Root Directory** to `web`. Keep the framework preset as **Other** and do not set a build command. Vercel serves `index.html`, `styles.css`, and `app.js` as static assets and deploys each file in `api/` as a Python Function.

The repository's root `requirements.txt` includes CPU PyTorch for model training. The Vercel root is intentionally `web/`, so this preview does not install or bundle PyTorch. The API uses Python's standard library only.

## Routes

- `/` — Android-friendly PyMind-A dashboard.
- `/api/health` — Python API health check.
- `/api/generate` — POST JSON `{"prompt":"Create a portfolio website"}`; returns generated HTML, CSS and JavaScript.
- `/api/generate?format=zip` — POST the same JSON to download a ZIP.

The generated preview runs in a sandboxed iframe. User prompt text is HTML-escaped before insertion. Requests are limited to 1,200 prompt characters and 10 KB request bodies.

## Important scope

This is a preview of the website builder, not the full TinyGPT chat application. Model training remains in GitHub Actions. TinyGPT checkpoint inference, image generation, video generation and production 3D model generation are not exposed by these lightweight serverless routes.

## Local smoke test

From the repository root:

```bash
python -m unittest tests.test_vercel_preview -v
```

For local browser preview, install Vercel CLI and run `vercel dev` from this `web/` directory.
