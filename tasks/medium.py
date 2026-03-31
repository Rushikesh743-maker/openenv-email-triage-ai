class MediumTask:

    def __init__(self):
        self.done = False
        self.email = {
            "id": 1,
            "sender": "hr@company.com",
            "subject": "Interview",
            "body": "Confirm your availability",
        }

    def reset(self):
        self.done = False
        return self.get_state()

    def get_state(self):
        return {"inbox": [self.email], "current_email_index": 0}

    def apply_action(self, action):
        reply = (action.reply_text or "").lower()
        score = 0
        if "thank" in reply: score += 2
        if "confirm" in reply: score += 2
        if len(reply) > 10: score += 1
        self.done = True
        return {"score": score}

    def compute_reward(self, result):
        return result["score"]

    def is_done(self):
        return self.done
