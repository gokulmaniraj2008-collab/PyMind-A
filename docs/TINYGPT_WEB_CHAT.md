# TinyGPT web chat integration

## Architecture

The Vercel site stays lightweight. Its \`/api/chat\` Python Function forwards a bounded chat request to a separate Python inference server. The inference server loads the actual PyTorch checkpoint through the existing \`LocalChatEngine\` implementation.

- Browser: \`web/\` chat panel.
- Vercel: \`web/api/chat.py\` proxy (standard library only).
- Model runtime: \`python -m pymind.inference_server\`.
- Model file: \`checkpoints/tiny_gpt.pt\` (created by the training workflow).

Do not add PyTorch to the Vercel \`web/\` root. The previous deployment demonstrated that including the root training dependencies exceeds the Vercel function bundle limit.

## Run the actual model server

From the repository root, install \`requirements.txt\`, download a fresh \`pymind-a-trained-model\` artifact from the successful GitHub Actions training run, and extract \`checkpoints/tiny_gpt.pt\` into the repository's \`checkpoints/\` folder. Then run:

\`\`\`bash
python -m pymind.inference_server
\`\`\`

The server listens on port 8000 by default. Set \`PYMIND_CHECKPOINT\` to use another checkpoint path. Set \`PYMIND_INFERENCE_TOKEN\` to require bearer-token authentication; use the same token in Vercel.

## Connect Vercel

In the Vercel project settings, add these server-side environment variables for Preview and Production:

- \`PYMIND_INFERENCE_URL\`: public HTTPS base URL of the running inference server (no trailing slash; the proxy appends \`/chat\`).
- \`PYMIND_INFERENCE_TOKEN\`: optional shared secret; set it on both services if authentication is enabled.

Redeploy after setting environment variables. Never put the token in browser JavaScript. The inference host must be reachable over HTTPS from Vercel and must remain running. A sleeping or stopped free host may make chat temporarily unavailable.

## API

\`POST /api/chat\` with JSON:

\`\`\`json
{"message":"Explain Python lists","history":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}
\`\`\`

The response contains \`reply\` and \`model\`. Requests are limited to 1,200 characters and the last eight history turns. If no inference URL is configured, the API returns HTTP 503 rather than inventing a model answer.

## Model limitations

TinyGPT is an educational, small byte-level next-token model trained on a small text corpus. It is not ChatGPT-level and may produce incomplete or incorrect responses, including in Tamil. This integration exposes the real checkpoint; it does not claim that training has made it a high-quality general assistant.
