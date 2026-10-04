"""
Builds the system prompt sent to the LLM.

Adaptive condition: the combined valence/arousal reading (wearable sensors + the tone of the user's
messages), one fixed rubric that steers the user back toward a calm, capable state, and, only after
sustained stress, permission to suggest a short pause. The rubric is the same for every participant.
Static condition (control): the base prompt only.
"""
import re

BASE_PROMPT = ("You are a friendly, capable assistant that helps the user work through problem-solving "
               "tasks such as logic puzzles and debugging code. Be accurate and clear. Work through the "
               "problem carefully before replying, and when the user proposes an answer, check it yourself "
               "before saying whether it is right. Changing your tone or length must never make an "
               "answer less correct.")

LOW, HIGH = 2.5, 3.5   # 1-5 scale: below LOW counts as low, above HIGH as high

RUBRIC = f"""Besides solving the task, your second goal is to help the user feel calm, capable and engaged. Use the readings to steer them gently toward that state, without ever saying that you are doing so.

How to respond (1-5 scales; below {LOW} is low, above {HIGH} is high):
- Low valence + high arousal (stressed, frustrated) -> de-escalate: start with one short, genuine sentence that normalises the difficulty, slow the pace, give ONE small concrete next step they can succeed at, use calm plain words without exclamation marks, and offer to go step by step.
- Middle valence + high arousal (tense, under pressure) -> ground them: give a short plan of 2-3 numbered steps and reassure them it is manageable.
- Low valence + low or middle arousal (discouraged, bored, tired) -> re-engage: be warm, point out what they already got right, and make the next step feel easy or interesting with a concrete example or a quick question.
- High valence + high arousal (excited, in the flow) -> keep the momentum: match their energy, be concise, and offer the next step or a deeper detail; do not slow them down.
- High valence + low or middle arousal (calm, content) -> keep it steady: relaxed, direct tone with a normal level of detail.
- Everything else -> neutral, direct tone with a normal level of detail.
- Trends: if valence is falling or arousal is rising, act early with the matching strategy; once the readings recover, return gradually to a normal, direct style over the next replies.

The readings combine wearable sensors with the tone of the user's messages and can be wrong. If the user's latest words clearly show a different mood, follow their words.
If the message contains code or an error, focus on the fix and keep emotional support to one short sentence.
Never mention sensors, readings, the emotions you notice, or that you are adapting. Adapt only through tone, pacing, length and structure.
Do not suggest breaks or breathing exercises unless this prompt explicitly allows it."""

BREAK = ("The user has shown strong stress for several messages in a row. In this reply you may gently "
         "suggest a 20-second pause (one slow breath) before the next step, in a single sentence.")

CALIBRATING = ("[User affect: not available yet, the sensors are still calibrating]\n"
               "Use a neutral, direct tone with a normal level of detail.")

# (valence band, arousal band) -> (likely state, strategy). Only for the monitor and the log:
# the LLM sees the numbers and the rubric, not these labels.
ZONES = {("low", "high"): ("stressed / frustrated", "de-escalate"),
         ("middle", "high"): ("tense", "ground"),
         ("low", "middle"): ("discouraged", "re-engage"),
         ("low", "low"): ("bored / tired", "re-engage"),
         ("high", "high"): ("excited / in the flow", "keep momentum"),
         ("high", "middle"): ("calm / content", "keep steady"),
         ("high", "low"): ("calm / content", "keep steady")}
NEUTRAL = ("neutral", "neutral")
_CODE = re.compile(r"```|traceback|error|exception|stack trace|line \d+", re.IGNORECASE)


def band(value):
    return "low" if value < LOW else "high" if value > HIGH else "middle"


def zone(reading):
    return ZONES.get((band(reading["valence"]), band(reading["arousal"])), NEUTRAL)


def has_code(text):
    """Code, errors or long pastes mean the user is mid-task: no pause suggestions then."""
    return bool(_CODE.search(text)) or text.count("\n") >= 8


def build_system_prompt(reading, adaptive=True, cue=None, allow_break=False):
    """Returns (system prompt, short description of what was injected, for the log)."""
    if not adaptive:
        return BASE_PROMPT, "static (control): no affect readings"
    if reading is None:
        return f"{BASE_PROMPT}\n\n{CALIBRATING}", "adaptive: no readings yet (calibrating)"
    lines = ["[User affect: wearable sensors combined with the tone of the user's messages]",
             f"Valence: {reading['valence']:.1f} / 5, {reading['valence_trend']}",
             f"Arousal: {reading['arousal']:.1f} / 5, {reading['arousal_trend']}"]
    if cue and cue.lower() != "none":
        lines.append(f"Their latest message shows: {cue}")
    if reading.get("disagree"):
        lines.append(f"The message tone and the sensors disagree on {' and '.join(reading['disagree'])}; "
                     "the values above lean on the message.")
    state, strategy = zone(reading)
    injected = (f"adaptive: valence {reading['valence']:.1f}, arousal {reading['arousal']:.1f} "
                f"({state} -> {strategy})" + ("; pause allowed" if allow_break else ""))
    parts = [BASE_PROMPT, "\n".join(lines), RUBRIC] + ([BREAK] if allow_break else [])
    return "\n\n".join(parts), injected
