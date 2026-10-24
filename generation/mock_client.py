"""Offline stand-in for the LLM (tests / demo without Ollama)."""
import re


class MockClient:
    model = "mock"

    def is_up(self):
        return True

    def generate(self, prompt, temperature=None, num_predict=120):
        verbs = re.search(r"verbs: ([^\n]+)", prompt).group(1).split(", ")
        ctx = re.search(r'"""\n(.*?)\n"""', prompt, re.S).group(1)
        topic = " ".join(ctx.split()[:12])
        return f"{verbs[0].strip().capitalize()} the main idea in: {topic}?"
