---
title: OpenEnv Email Triage AI
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage OpenEnv (Final)

## Real-world AI environment for email automation.

### Tasks
- Easy: classification
- Medium: reply generation
- Hard: workflow automation

### Run
docker build -t env .
docker run -p 7860:7860 env

### API
POST /reset
POST /step

### Inference
python inference.py
