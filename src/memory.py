class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def add(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in self.history
        )
