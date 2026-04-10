import os
from openai import OpenAI
from env.environment import EmailEnv
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.action import Action

# ---------------- SAFE ENV SETUP ----------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")  # REQUIRED
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not HF_TOKEN:
    raise Exception("❌ HF_TOKEN missing in environment variables")

# ---------------- LLM CLIENT ----------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ---------------- AGENT ----------------

def choose_action(obs):
    prompt = f"""
You are an AI email triage agent.

You MUST choose ONLY ONE word from:
delete, reply, read, skip

Rules:
- spam / promotions → delete
- urgent / important → reply
- normal → read

Email:
{obs}

Return ONLY one word.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only one word: delete, reply, read, skip."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )

        action = response.choices[0].message.content.strip().lower()
        action = action.replace(".", "").replace("\n", "").strip()

        if action not in ["delete", "reply", "read", "skip"]:
            action = "read"

        return Action(action_type=action, email_id=0)

    except Exception as e:
        print(f"[LLM ERROR] {e}", flush=True)
        return Action(action_type="read", email_id=0)

# ---------------- RUN LOOP ----------------

def run(task_name, env):
    obs = env.reset()
    total = 0
    steps = 0

    print(f"[START] task={task_name}", flush=True)

    for i in range(10):
        action = choose_action(obs)

        obs, reward, done, _ = env.step(action)

        steps += 1
        total += reward.value

        print(f"[STEP] step={steps} reward={reward.value}", flush=True)

        if done:
            break

    print(f"[END] task={task_name} score={total} steps={steps}", flush=True)

# ---------------- MAIN ----------------

def main():
    try:
        run("easy", EmailEnv(EasyTask()))
        run("medium", EmailEnv(MediumTask()))
        run("hard", EmailEnv(HardTask()))
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)

if __name__ == "__main__":
    main()
