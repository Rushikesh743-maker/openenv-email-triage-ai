import os
import json
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI()

# ---------------- LLM PROXY CLIENT (MANDATORY) ----------------
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

# ---------------- CORE LLM LOGIC ----------------

def analyze_email(text: str):
    prompt = f"""
You are an AI email triage system.

Classify the email into:
- Spam
- Important
- Normal

Decide action:
- Spam → Delete 🚨
- Important → Reply 📩
- Normal → Mark as Read ✔

Also write a short professional reply.

Return ONLY valid JSON:
{{
  "category": "...",
  "action": "...",
  "reply": "...",
  "confidence": 0.0
}}

Email:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    result = response.choices[0].message.content

    try:
        return json.loads(result)
    except:
        return {
            "category": "Normal",
            "action": "Mark as Read",
            "reply": result,
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


# ---------------- REQUIRED ENDPOINTS (OPENENV / HACKATHON) ----------------

@app.post("/reset")
def reset():
    return {"status": "reset done"}


@app.get("/health")
def health():
    return {"status": "running"}
