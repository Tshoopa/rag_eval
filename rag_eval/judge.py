"""
LLM-as-a-judge client.

Thin wrapper around an OpenAI-compatible endpoint (DeepSeek by default) used
by all metrics. Runs at temperature 0 for deterministic, reproducible
verdicts, and defensively extracts JSON from model output.
"""
import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMJudge:
    """Deterministic judge that returns parsed JSON verdicts from an LLM."""

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        # temperature=0 keeps judgments reproducible across runs.
        self.temperature = temperature

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        """Send a single chat completion and return the raw text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def ask_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a prompt and parse a JSON object out of the response."""
        raw = self.ask(system_prompt, user_prompt)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract a JSON object from model output, tolerating markdown fences."""
        # Strip ```json ... ``` fences if present.
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        # Grab the outermost {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Fail safe: surface the raw text instead of raising.
        return {"error": "could not parse JSON", "raw": text}