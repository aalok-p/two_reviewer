---
title: Kernel Writer
aemoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
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

## Hugging Face Space setup

Set the OpenAI key in Space **Settings → Variables and secrets** as:

- `OPENAI_API_KEY`

Optional:

- `MODEL_NAME` (default: `gpt-4`)
- `API_BASE_URL` (default: `https://api.openai.com/v1`)
