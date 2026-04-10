import os
from openai import OpenAI
from env.environment import EmailEnv
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.action import Action

# ---------------- LLM PROXY CLIENT ----------------
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

# ---------------- LLM-BASED AGENT ----------------

def choose_action(obs):
    prompt = f"""
You are an AI email triage agent.

You MUST choose ONE action:
- delete
- reply
- read
- skip

Rules:
- Spam / promotions → delete
- Important / urgent → reply
- Normal → read

Return ONLY one word.

Email:
{obs}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict action selector."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    action = response.choices[0].message.content.strip().lower()

    # safety fallback (just in case model returns weird output)
    if action not in ["delete", "reply", "read", "skip"]:
        action = "read"

    return Action(action_type=action, email_id=0)


# ---------------- RUN LOOP ----------------

def run(task_name, env):
    obs = env.reset()
    total = 0
    steps = 0

    print(f"[START] task={task_name}", flush=True)

    for i in range(10):
        try:
            action = choose_action(obs)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            action = Action(action_type="skip", email_id=0)

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
