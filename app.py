import os
import json
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI()

# ---------------- ENV CONFIG ----------------

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ---------------- SAFE LLM PING (REQUIRED) ----------------

def ping_llm():
    try:
        if not API_BASE_URL or not API_KEY:
            print("[PING SKIPPED] Missing API_BASE_URL or API_KEY", flush=True)
            return

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )

        print("[LLM PING SUCCESS]", flush=True)

    except Exception as e:
        print("[LLM PING FAILED]", e, flush=True)

# Run at startup
@app.on_event("startup")
def startup_event():
    ping_llm()

# ---------------- SAFE CLIENT (OPTIONAL USE) ----------------

client = None

if API_BASE_URL and API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
    except Exception as e:
        print("[CLIENT INIT ERROR]", e, flush=True)
        client = None

# ---------------- CORE LOGIC ----------------

def analyze_email(text: str):
    text_lower = text.lower()

    # -------- RULE-BASED (PRIMARY SAFE) --------
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

    # -------- OPTIONAL LLM --------
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
            body {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                text-align: center;
            }
            .container {
                width: 60%;
                margin: auto;
                margin-top: 60px;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
            }
            textarea {
                width: 100%;
                padding: 15px;
                border-radius: 10px;
                border: none;
                margin-top: 20px;
                font-size: 16px;
            }
            button {
                margin-top: 20px;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                background: #22c55e;
                color: white;
                font-size: 16px;
                cursor: pointer;
                transition: 0.3s;
            }
            button:hover {
                background: #16a34a;
            }
            h1 {
                font-size: 32px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Email Assistant</h1>
            <p>Paste your email and let AI decide the best action</p>
            <form action="/predict" method="post">
                <textarea name="text" rows="6" placeholder="Paste your email here..."></textarea>
                <br>
                <button type="submit">Analyze Email</button>
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
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #0f172a, #1e293b);
                color: white;
                text-align: center;
            }}
            .container {{
                width: 60%;
                margin: auto;
                margin-top: 50px;
            }}
            .card {{
                background: rgba(255,255,255,0.05);
                padding: 20px;
                margin: 20px 0;
                border-radius: 12px;
                box-shadow: 0 0 15px rgba(0,0,0,0.5);
            }}
            .title {{
                font-size: 28px;
                margin-bottom: 20px;
            }}
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                background: #3b82f6;
                color: white;
                cursor: pointer;
            }}
            .btn:hover {{
                background: #2563eb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">📧 Email Analysis Result</h1>
            <div class="card">
                <h2>📊 Category</h2>
                <p>{result.get('category')}</p>
                <p>Confidence: {result.get('confidence')}</p>
            </div>
            <div class="card">
                <h2>⚡ Suggested Action</h2>
                <p>{result.get('action')}</p>
            </div>
            <div class="card">
                <h2>✍️ Suggested Reply</h2>
                <p>{result.get('reply')}</p>
            </div>
            <a href="/"><button class="btn">🔙 Back</button></a>
        </div>
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
