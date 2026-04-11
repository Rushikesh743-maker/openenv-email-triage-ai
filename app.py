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

# Do NOT crash if token missing
if not HF_TOKEN:
    print("[WARNING] No API key found, running in fallback mode", flush=True)

# ---------------- SAFE CLIENT INIT ----------------

client = None

if HF_TOKEN:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception as e:
        print("[CLIENT INIT ERROR]", e, flush=True)
        client = None

# ---------------- CORE LOGIC ----------------

def analyze_email(text: str):
    text_lower = text.lower()

    # -------- RULE-BASED (PRIMARY, SAFE) --------
    if any(word in text_lower for word in ["spam", "offer", "buy now", "free", "discount"]):
        return {
            "category": "Spam",
            "action": "Delete",
            "reply": "",
            "confidence": 0.9
        }

    if any(word in text_lower for word in ["urgent", "asap", "important"]):
        return {
            "category": "Important",
            "action": "Reply",
            "reply": "I will get back to you shortly.",
            "confidence": 0.9
        }

    # -------- OPTIONAL LLM (only if available) --------
    if client:
        try:
            prompt = f"""
Return ONLY valid JSON:
{{
  "category": "Spam | Important | Normal",
  "action": "Delete | Reply | Mark as Read",
  "reply": "short professional reply",
  "confidence": 0.0
}}
Email:
{text}
"""

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a strict JSON generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()

            try:
                return json.loads(result)
            except:
                cleaned = result.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)

        except Exception as e:
            print(f"[LLM ERROR] {e}", flush=True)

    # -------- SAFE DEFAULT --------
    return {
        "category": "Normal",
        "action": "Mark as Read",
        "reply": "Noted. Thank you.",
        "confidence": 0.5
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
            <h1>🤖 AI Email Agent</h1>
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
    try:
        result = analyze_email(text)
    except Exception as e:
        print("[PREDICT ERROR]", e, flush=True)
        result = {
            "category": "Normal",
            "action": "Mark as Read",
            "reply": "Error occurred.",
            "confidence": 0.0
        }

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

# ---------------- REQUIRED ROUTES ----------------

@app.post("/reset")
def reset():
    return {"status": "reset done"}

@app.get("/health")
def health():
    return {"status": "running"}
