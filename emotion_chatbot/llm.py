"""Chat backends: OpenAI (key read from .env) and an offline test mode."""
import os

from openai import BadRequestError, OpenAI

from config import OPENAI_MODEL, OPENAI_REASONING_EFFORT, OPENAI_TEMPERATURE

MAX_REPLY_TOKENS = 8000   # reasoning models spend part of this budget thinking before they write
_REJECTED = {}            # model -> optional parameters it refused, so they are not sent again


def api_key_present():
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def chat_create(client, optional, **kwargs):
    """chat.completions.create, dropping optional parameters the model rejects: reasoning models
    refuse `temperature`, older chat models refuse `reasoning_effort`."""
    rejected = _REJECTED.setdefault(kwargs["model"], set())
    optional = {k: v for k, v in optional.items() if k not in rejected}
    while True:
        try:
            return client.chat.completions.create(**kwargs, **optional)
        except BadRequestError as e:
            bad = next((k for k in optional if k in str(e)), None)
            if bad is None:
                raise
            rejected.add(bad)
            optional.pop(bad)


class OpenAIChat:
    def __init__(self, model=OPENAI_MODEL):
        if not api_key_present():
            raise RuntimeError("OPENAI_API_KEY is empty. Add your key to emotion_chatbot/.env "
                               "and restart the app.")
        self.client = OpenAI()   # reads OPENAI_API_KEY from the environment
        self.name = f"OpenAI {model}"
        self.model = model

    def stream(self, messages):
        optional = {"reasoning_effort": OPENAI_REASONING_EFFORT}
        if OPENAI_TEMPERATURE:
            optional["temperature"] = float(OPENAI_TEMPERATURE)
        response = chat_create(self.client, optional, model=self.model, messages=messages, stream=True,
                               max_completion_tokens=MAX_REPLY_TOKENS)
        wrote = False
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                wrote = True
                yield chunk.choices[0].delta.content
        if not wrote:
            raise RuntimeError("the model returned an empty reply (it may have spent its whole token "
                               "budget reasoning). Please send the message again.")


class TestChat:
    """No API call: shows which readings were injected. For trying the app without a key."""
    name = "test mode (no API calls)"

    def stream(self, messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        injected = [line for line in system.splitlines() if line.startswith(("Valence:", "Arousal:", "[User affect"))]
        yield ("**[test reply]** The model would receive:\n\n"
               + ("\n".join(f"- {line}" for line in injected) or "- no affect readings (static condition)")
               + f"\n\nYou said: {user}")
