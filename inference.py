import os
from openai import OpenAI
from env.environment import EmailEnv
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.action import Action

# ---------------- SAFE ENV SETUP ----------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional now
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = None

# Only initialize client if token exists
if HF_TOKEN:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception as e:
        print(f"[INIT ERROR] {e}", flush=True)
        client = None
else:
    print("[WARNING] HF_TOKEN not found, running in rule-based mode", flush=True)

# ---------------- AGENT ----------------

def choose_action(obs):
    text = str(obs).lower()

    # -------- RULE-BASED (SAFE & PRIMARY) --------
    if any(word in text for word in ["spam", "offer", "buy now", "free", "discount"]):
        return Action(action_type="delete", email_id=0)

    if any(word in text for word in ["urgent", "asap", "immediately", "important"]):
        return Action(action_type="reply", email_id=0)

    if any(word in text for word in ["meeting", "schedule", "update"]):
        return Action(action_type="read", email_id=0)

    # -------- LLM FALLBACK (OPTIONAL) --------
    if client:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Return only one word: delete, reply, read, skip."},
                    {"role": "user", "content": text}
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

    # -------- DEFAULT SAFE ACTION --------
    return Action(action_type="read", email_id=0)

# ---------------- RUN LOOP ----------------

def run(task_name, env):
    try:
        obs = env.reset()
    except Exception as e:
        print(f"[ENV RESET ERROR] {e}", flush=True)
        return

    total = 0
    steps = 0

    print(f"[START] task={task_name}", flush=True)

    for i in range(10):
        try:
            action = choose_action(obs)
            obs, reward, done, _ = env.step(action)

            steps += 1
            total += getattr(reward, "value", 0)

            print(f"[STEP] step={steps} reward={getattr(reward, 'value', 0)}", flush=True)

            if done:
                break

        except Exception as e:
            print(f"[STEP ERROR] {e}", flush=True)
            break

    print(f"[END] task={task_name} score={total} steps={steps}", flush=True)

# ---------------- MAIN ----------------

def main():
    try:
        run("easy", EmailEnv(EasyTask()))
        run("medium", EmailEnv(MediumTask()))
        run("hard", EmailEnv(HardTask()))
    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)

if __name__ == "__main__":
    main()
