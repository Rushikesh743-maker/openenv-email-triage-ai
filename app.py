from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
print("🔥 APP FILE LOADED SUCCESSFULLY")
app = FastAPI()

# ---------- Models ----------
class Email(BaseModel):
    subject: str
    type: str  # spam / important / normal

class Action(BaseModel):
    action_type: str  # classify / reply / skip
    label: str = None


# ---------- Environment State ----------
emails = [
    {"subject": "Meeting tomorrow", "type": "important"},
    {"subject": "Win a free iPhone", "type": "spam"},
    {"subject": "Project update", "type": "normal"},
]

current_index = 0
total_reward = 0


# ---------- Root ----------
@app.get("/")
def home():
    return {"message": "OpenEnv Email Triage AI is running 🚀"}


# ---------- Reset ----------
@app.post("/reset")
def reset():
    global current_index, total_reward
    current_index = 0
    total_reward = 0

    return {
        "observation": emails[current_index],
        "reward": 0.0,
        "done": False,
        "info": {}
    }


# ---------- Step ----------
@app.post("/step")
def step(action: Action):
    global current_index, total_reward

    if current_index >= len(emails):
        return {
            "observation": None,
            "reward": 0.0,
            "done": True,
            "info": {"message": "All emails processed"}
        }

    current_email = emails[current_index]

    # 🎯 Reward Logic
    reward = 0.0

    if action.action_type == "classify":
        if action.label == current_email["type"]:
            reward = 1.0
        else:
            reward = -0.5

    elif action.action_type == "skip":
        reward = -0.2

    elif action.action_type == "reply":
        if current_email["type"] == "important":
            reward = 1.0
        else:
            reward = -0.3

    total_reward += reward
    current_index += 1

    done = current_index >= len(emails)

    next_obs = emails[current_index] if not done else None

    return {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "info": {
            "total_reward": total_reward
        }
    }
