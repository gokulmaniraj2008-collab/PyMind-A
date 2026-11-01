# PyMind-A Vercel preview

This folder is a separate, lightweight Vercel project root. It contains a mobile-friendly static UI and Python serverless functions that use only the Python standard library.

## Vercel project settings

Set **Root Directory** to \`web\`, framework preset to **Other**, and leave the build command empty. This avoids bundling the repository-root PyTorch training dependencies.

## Routes

- \`/\` — Android-friendly dashboard with website builder and TinyGPT chat UI.
- \`/api/health\` — Python API health check.
- \`/api/generate\` — prompt-to-website generator.
- \`/api/generate?format=zip\` — ZIP export.
- \`/api/chat\` — secure proxy to a separately hosted TinyGPT inference service.

See [TinyGPT web chat setup](../docs/TINYGPT_WEB_CHAT.md) for model-server and Vercel environment configuration.

## Scope

Training remains in GitHub Actions. The Vercel API does not install PyTorch or contain model weights. TinyGPT inference requires a running Python model server and a trained checkpoint. If the model service is not configured, the chat UI clearly reports that it is offline.


## Chat behavior

The experimental chat first checks the bilingual training corpus for an exact or closely matching beginner question. If a match is found, it returns the stored answer; otherwise it uses the local TinyGPT generator. This hybrid approach makes known starter questions more reliable while the small generative model remains experimental and is not a ChatGPT-scale assistant.
