"""Paths and settings shared by training and the app. Secrets (the OpenAI key) live in .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

KEMOCON_ROOT = Path(os.environ.get("KEMOCON_ROOT") or APP_DIR.parent / "K EmoCon")
MODEL_PATH = APP_DIR / "models" / "va_model.joblib"
LOG_DIR = APP_DIR / "logs"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
OPENAI_TEMPERATURE = os.environ.get("OPENAI_TEMPERATURE")   # unset = the model's default
# Reasoning models only: "low" keeps replies quick for a live chat; ignored by models without reasoning.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT") or "low"

TEXT_AFFECT_MODEL = os.environ.get("TEXT_AFFECT_MODEL") or OPENAI_MODEL   # rates each message's tone

WINDOW_SEC = 10       # each estimate uses the last 10 s of signal
STEP_SEC = 5          # a new estimate every 5 s (K-EmoCon's annotation rate)
BASELINE_SEC = 120    # the first 2 minutes of a session are the person's baseline
SMOOTH_SEC = 30       # readings sent to the LLM are averaged over the last 30 s

# Combining sensors with the tone of the user's messages
TEXT_WEIGHT = {"valence": 0.8, "arousal": 0.5}  # share taken from the message at full confidence
TEXT_DECAY = 0.6        # per message, how much of an old message estimate's confidence is kept
DISAGREE = 1.5          # gap (1-5 scale) between message and sensors worth telling the LLM about
STRESS_TURNS = 2        # consecutive stressed messages before a short pause may be suggested
BREAK_GAP_SEC = 600     # ... and at most once every 10 minutes
