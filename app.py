import os
import json
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI()

# ---------------- SAFE ENV CONFIG ----------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not HF_TOKEN:
    raise Exception("❌ Missing HF_TOKEN / API_KEY in environment variables")

# ---------------- HF CLIENT ----------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ---------------- CORE LOGIC ----------------

def analyze_email(text: str):
    prompt = f"""
You are an AI email assistant.

Return ONLY valid JSON (no markdown, no explanation).

Schema:
{{
  "category": "Spam | Important | Normal",
  "action": "Delete | Reply | Mark as Read",
  "reply": "short professional reply",
  "confidence": 0.0
}}

Rules:
- Spam emails → Spam + Delete
- Urgent emails → Important + Reply
- Others → Normal + Mark as Read

Email:
{text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=250
        )

        result = response.choices[0].message.content.strip()

        # ---------------- SAFE JSON PARSING ----------------
        try:
            return json.loads(result)
        except:
            cleaned = result.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return {
            "category": "Normal",
            "action": "Mark as Read",
            "reply": "Unable to generate response at the moment.",
            "confidence": 0.0
        }

# ---------------- UI ----------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <style>
            body { font-family: Arial; background:#0f172a; color:white; text-align:center; }
            .box { width:60%; margin:auto; margin-top:50px; }
            textarea { width:100%; padding:12px; border-radius:10px; }
            button { padding:10px 20px; border:none; border-radius:8px; background:#22c55e; color:white; cursor:pointer; }
            .card { background:#1e293b; padding:15px; border-radius:10px; margin-top:20px; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🤖 AI Email Agent (LLM Powered)</h1>
            <form action="/predict" method="post">
                <textarea name="text" rows="6" placeholder="Paste your email here..."></textarea><br><br>
                <button type="submit">Analyze</button>
            </form>
        </div>
    </body>
    </html>
    """

# ---------------- PREDICT ----------------

@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    result = analyze_email(text)

    return f"""
    <html>
    <body style="font-family:Arial;background:#0f172a;color:white;text-align:center;">
        <h1>📧 AI Email Result</h1>

        <div style="background:#1e293b;padding:20px;margin:20px;border-radius:10px;">
            <h2>📊 Category: {result.get('category')}</h2>
            <p>Confidence: {result.get('confidence')}</p>
        </div>

        <div style="background:#1e293b;padding:20px;margin:20px;border-radius:10px;">
            <h3>⚡ Action</h3>
            <p>{result.get('action')}</p>
        </div>

        <div style="background:#1e293b;padding:20px;margin:20px;border-radius:10px;">
            <h3>✍️ Reply</h3>
            <p>{result.get('reply')}</p>
        </div>

        <a href="/"><button style="padding:10px 20px;">🔙 Back</button></a>
    </body>
    </html>
    """

# ---------------- HACKATHON REQUIRED ROUTES ----------------

@app.post("/reset")
def reset():
    return {"status": "reset done"}

@app.get("/health")
def health():
    return {"status": "running"}
