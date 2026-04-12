---
title: Kernel Writer
aemoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Kernel Writer

CUDA kernel optimization 

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

OpenEnv API is served at `/reset`, `/step`, `/state`, and the Gradio UI is at `/ui`.

## Hugging Face Space setup

Set the OpenAI key in Space **Settings → Variables and secrets** as:

- `OPENAI_API_KEY`

Optional:

- `MODEL_NAME` (default: `gpt-5.4`)
- `API_BASE_URL` (default: `https://api.openai.com/v1`)

## Submission validation

Use the validator script with the **runtime URL**:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://aaloksan-kernel.hf.space .
```
