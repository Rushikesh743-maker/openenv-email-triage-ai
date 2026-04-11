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
    <body style="font-family:Arial;background:#0f172a;color:white;text-align:center;">
        <h1>🤖 AI Email Agent</h1>
        <form action="/predict" method="post">
            <textarea name="text" rows="6" style="width:60%;padding:10px;"></textarea><br><br>
            <button type="submit">Analyze</button>
        </form>
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
        <p><b>Category:</b> {result.get('category')}</p>
        <p><b>Action:</b> {result.get('action')}</p>
        <p><b>Reply:</b> {result.get('reply')}</p>
        <p><b>Confidence:</b> {result.get('confidence')}</p>
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
