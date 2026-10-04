"""
Reads the emotional tone of the user's latest message with a small OpenAI call.
Returns valence/arousal on the same 1-5 scale as the sensor model, plus a confidence and a short cue.
"""
import json

from openai import OpenAI

from config import TEXT_AFFECT_MODEL, TEXT_DECAY
from llm import chat_create

SYSTEM = """You rate the emotional state of a person chatting with a problem-solving assistant, from their LATEST message. Earlier messages are context only.
Return JSON only: {"valence": number, "arousal": number, "confidence": number, "cue": string}
- valence 1-5: 1 = very negative (frustrated, upset, discouraged), 3 = neutral, 5 = very positive (pleased, relieved, excited)
- arousal 1-5: 1 = very calm or low-energy (bored, tired), 3 = normal, 5 = very activated (stressed, agitated, excited)
- confidence 0-1: how much the message REVEALS about their feelings. This is not how sure you are of your rating: a short, polite or purely factual message reveals little, so give it 0.2 or less and values near 3 (e.g. "ok", "thanks", "what is 15% of 80?", "can you help me with a puzzle?"). Use 0.7 or more only when emotion is clearly expressed.
- cue: at most 8 words naming what shows the emotion (e.g. "says nothing works, repeated question"), or "none"."""


class TextAffect:
    def __init__(self, model=TEXT_AFFECT_MODEL):
        self.client = OpenAI()     # reads OPENAI_API_KEY from the environment
        self.model = model

    def rate(self, messages):
        """messages: the chat so far, ending with the user's latest message."""
        context = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in messages[-7:-1]) or "(none)"
        prompt = (f"Earlier conversation (context only):\n{context}\n\n"
                  f"Latest user message to rate:\n{messages[-1]['content'][:2000]}")
        response = chat_create(self.client, {"temperature": 0, "reasoning_effort": "low"},
                               model=self.model, response_format={"type": "json_object"},
                               max_completion_tokens=3000,
                               messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        d = json.loads(response.choices[0].message.content)
        return {"valence": _clip(d["valence"], 1, 5), "arousal": _clip(d["arousal"], 1, 5),
                "confidence": _clip(d.get("confidence", 0.5), 0, 1), "cue": str(d.get("cue", ""))[:80]}


def _clip(x, lo, hi):
    return round(min(hi, max(lo, float(x))), 2)


class TextTracker:
    """Keeps the most informative recent message estimate. A new message replaces it only if it says
    at least as much as the old estimate, whose confidence fades with every message; so "ok" after an
    angry message does not wipe the anger out at once."""

    def __init__(self):
        self.current = None

    def update(self, estimate):
        if self.current:
            self.current = {**self.current, "confidence": round(self.current["confidence"] * TEXT_DECAY, 2),
                            "carried_over": True}
        if estimate and (self.current is None or estimate["confidence"] >= self.current["confidence"]):
            self.current = {**estimate, "carried_over": False}
        return self.current
