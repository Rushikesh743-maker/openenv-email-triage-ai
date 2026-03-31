---
title: openenv-email-triage-ai
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv Email Triage AI

AI-powered OpenEnv environment that simulates real-world email triage tasks including classification, response generation, and workflow automation.

## Features
- Email classification (spam, important, normal)
- Automated reply generation
- Multi-step workflow simulation

## API Endpoints
- POST /reset
- POST /step

## Run Locally
docker build -t env .
docker run -p 7860:7860 env
