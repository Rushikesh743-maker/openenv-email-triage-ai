import json

class EasyTask:

    def __init__(self):
        self.index = 0
        self.done = False
        with open("data/emails.json") as f:
            self.emails = json.load(f)

    def reset(self):
        self.index = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return {"inbox": self.emails, "current_email_index": self.index}

    def apply_action(self, action):
        email = self.emails[self.index]
        correct = action.classification == email["label"]

        self.index += 1
        if self.index >= len(self.emails):
            self.done = True

        return {"correct": correct}

    def compute_reward(self, result):
        reward = 2 if result["correct"] else -1
        reward -= 0.1
        if self.done:
            reward += 1
        return reward

    def is_done(self):
        return self.done
