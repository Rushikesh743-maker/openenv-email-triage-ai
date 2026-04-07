import json
from env.environment import EmailEnv
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.action import Action


# 🔹 Simple rule-based agent (no OpenAI)
def choose_action(obs):
    text = str(obs).lower()

    if any(word in text for word in ["free", "win", "offer", "money"]):
        return Action(action_type="delete", email_id=0)
    elif "urgent" in text or "important" in text:
        return Action(action_type="reply", email_id=0)
    else:
        return Action(action_type="read", email_id=0)


def run(task_name, env):
    obs = env.reset()
    total = 0
    steps = 0

    print(f"[START] task={task_name}", flush=True)

    for i in range(10):
        try:
            action = choose_action(obs)
        except:
            action = Action(action_type="skip", email_id=0)

        obs, reward, done, _ = env.step(action)

        steps += 1
        total += reward.value

        print(f"[STEP] step={steps} reward={reward.value}", flush=True)

        if done:
            break

    print(f"[END] task={task_name} score={total} steps={steps}", flush=True)


def main():
    try:
        run("easy", EmailEnv(EasyTask()))
        run("medium", EmailEnv(MediumTask()))
        run("hard", EmailEnv(HardTask()))
    except Exception as e:
        print(f"[ERROR] {str(e)}", flush=True)


if __name__ == "__main__":
    main()
