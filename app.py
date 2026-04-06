from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# ---------------- AI MODELS ----------------

# Better classifier
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# FLAN-T5
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# ---------------- REPLY GENERATOR ----------------
def generate_reply(text):
    prompt = f"""
You are a professional AI email assistant.
Write a polite, natural, and helpful reply to the email below.
The reply should be 4-6 sentences long and feel human.
Email:
{text}
Reply:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------- CORE LOGIC ----------------
def analyze_email(text):
    # classification
    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 2)

    if label == "POSITIVE":
        category = "Important"
    else:
        category = "Normal"

    # spam override
    if any(word in text.lower() for word in ["free", "win", "offer", "money"]):
        category = "Spam"

    # action
    if category == "Spam":
        action = "Delete 🚨"
    elif category == "Important":
        action = "Reply 📩"
    else:
        action = "Mark as Read ✔"

    reply = generate_reply(text)

    return category, action, reply, score


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


@app.post("/predict", response_class=HTMLResponse)
def predict(text: str = Form(...)):
    category, action, reply, score = analyze_email(text)

    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial; background:#0f172a; color:white; text-align:center; }}
            .box {{ width:60%; margin:auto; margin-top:50px; }}
            .card {{ background:#1e293b; padding:20px; border-radius:10px; margin-top:20px; }}
            button {{ padding:10px 20px; border:none; border-radius:8px; background:#22c55e; color:white; cursor:pointer; }}
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🤖 AI Email Agent</h1>
            <div class="card">
                <h2>📊 Category: {category}</h2>
                <p>Confidence: {score}</p>
            </div>
            <div class="card">
                <h3>⚡ Suggested Action</h3>
                <p>{action}</p>
            </div>
            <div class="card">
                <h3>✍️ AI Reply</h3>
                <p>{reply}</p>
            </div>
            <br>
            <a href="/"><button>🔙 Try Another</button></a>
        </div>
    </body>
    </html>
    """
    # ---------------- OPENENV REQUIRED ROUTES ----------------

@app.post("/reset")
def reset():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "running"}
