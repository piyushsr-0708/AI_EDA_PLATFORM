# planner.py
# The planner relies on the system instruction in prompts.py to guide the LLM through the linear steps.
class AgentPlanner:
    def __init__(self, steps):
        self.steps = steps
