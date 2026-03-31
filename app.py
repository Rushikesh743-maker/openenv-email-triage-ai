from fastapi import FastAPI

app = FastAPI()

# Root endpoint (for browser test)
@app.get("/")
def home():
    return {"message": "OpenEnv Email Triage AI is running 🚀"}

# Reset endpoint
@app.post("/reset")
def reset():
    return {
        "inbox": [
            {"subject": "Meeting tomorrow", "type": "important"},
            {"subject": "Win a free iPhone", "type": "spam"},
            {"subject": "Project update", "type": "normal"}
        ],
        "current_email_index": 0
    }

# Step endpoint
@app.post("/step")
def step(action: dict):
    return {
        "observation": "processed",
        "reward": 1.0,
        "done": False,
        "info": {}
    }
