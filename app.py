from fastapi import FastAPI
from env.environment import EmailEnv
from tasks.easy import EasyTask
from env.action import Action

app = FastAPI()
env = EmailEnv(EasyTask())

@app.post("/reset")
def reset():
    return env.reset().model_dump()

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)
    return {"observation": obs.model_dump(),"reward": reward.value,"done": done,"info": info}
