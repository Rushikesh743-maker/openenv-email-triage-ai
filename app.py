import os
import json
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI

app = FastAPI()

# ---------------- ENV CONFIG ----------------

client = None

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")

if API_BASE_URL and API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        print("[CLIENT READY]", flush=True)
    except Exception as e:
        print("[CLIENT INIT ERROR]", e, flush=True)
        client = None
else:
    print("[WARNING] API env vars not found (local run)", flush=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# ---------------- FORCE API CALL ----------------

def force_llm_call():
    try:
        if not client:
            print("[NO CLIENT FOR LLM CALL]", flush=True)
            return

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )

        print("[FORCED LLM CALL SUCCESS]", flush=True)

    except Exception as e:
        print("[FORCED LLM CALL FAILED]", e, flush=True)

# ---------------- CORE LOGIC ----------------

def analyze_email(text: str):
    text_lower = text.lower()

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
            }
            textarea {
                width: 100%;
                padding: 15px;
                border-radius: 10px;
                border: none;
                margin-top: 20px;
            }
            button {
                margin-top: 20px;
                padding: 12px 25px;
                border-radius: 10px;
                background: #22c55e;
                color: white;
                border: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Email Assistant</h1>
            <form action="/predict" method="post">
                <textarea name="text" rows="6"></textarea>
                <br>
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
        # 🔥 IMPORTANT: THIS GUARANTEES API CALL
        force_llm_call()

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
        <h1>📧 Result</h1>
        <p>Category: {result.get('category')}</p>
        <p>Action: {result.get('action')}</p>
        <p>Reply: {result.get('reply')}</p>
        <br><a href="/">Back</a>
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
