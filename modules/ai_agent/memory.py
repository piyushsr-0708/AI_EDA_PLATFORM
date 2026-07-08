# memory.py
# The memory is primarily handled by the Gemini SDK's chat session, which maintains the conversation history.
class AgentMemory:
    def __init__(self):
        self.history = []

    def get_history(self):
        return self.history
