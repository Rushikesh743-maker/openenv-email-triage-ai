import os, json
from openai import OpenAI
from env.environment import EmailEnv
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.action import Action

client = OpenAI(base_url=os.getenv("API_BASE_URL"), api_key=os.getenv("HF_TOKEN"))

SYSTEM_PROMPT = "You are an email assistant. Return only JSON."

def run(env):
    obs = env.reset()
    total = 0
    for _ in range(10):
        prompt = str(obs)
        try:
            res = client.chat.completions.create(
                model=os.getenv("MODEL_NAME"),
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
            )
            action = Action(**json.loads(res.choices[0].message.content))
        except:
            action = Action(action_type="skip", email_id=0)
        obs, reward, done, _ = env.step(action)
        total += reward.value
        if done: break
    return total

def main():
    results = {}
    for name, task in [("easy",EasyTask()),("medium",MediumTask()),("hard",HardTask())]:
        results[name] = run(EmailEnv(task))
    print(results)

if __name__ == "__main__":
    main()
