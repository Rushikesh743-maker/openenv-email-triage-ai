class HardTask:

    def __init__(self):
        self.steps = 0
        self.done = False

    def reset(self):
        self.steps = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return {"inbox": [], "current_email_index": self.steps}

    def apply_action(self, action):
        self.steps += 1
        if self.steps >= 3:
            self.done = True
        return {"progress": self.steps}

    def compute_reward(self, result):
        return float(result["progress"])

    def is_done(self):
        return self.done
