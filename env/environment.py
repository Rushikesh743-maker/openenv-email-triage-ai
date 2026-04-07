from env.observation import Observation, Email
from env.reward import Reward

class EmailEnv:

    def __init__(self, task):
        self.task = task
        self.state_data = None

    def reset(self):
        self.state_data = self.task.reset()
        return self.state()

    def step(self, action):
        result = self.task.apply_action(action)
        reward_value = self.task.compute_reward(result)
        done = self.task.is_done()
        self.state_data = self.task.get_state()
        return self.state(), Reward(value=reward_value, reason="step"), done, result

    def state(self):
        emails = [Email(**e) for e in self.state_data["inbox"]]
        return Observation(
            inbox=emails,
            current_email_index=self.state_data["current_email_index"]
        )
