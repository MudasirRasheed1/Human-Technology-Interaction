"""
Emotion-aware chatbot (HTI project).

Wearable signals -> valence/arousal model -> readings in the system prompt -> OpenAI -> reply.
For now the signals come from replaying K-EmoCon recordings in real time (or manual sliders for
Wizard-of-Oz testing); a live device can feed the same AffectEstimator later.

Run from this folder with the `torch` conda env:
    conda activate torch
    python train_va_model.py      # once: creates models/va_model.joblib
    streamlit run app.py
Your OpenAI key goes in .env (OPENAI_API_KEY=...).
"""
import json
import re
import time
import uuid

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from affect_model import AffectEstimator, Smoother, fuse
from config import (BASELINE_SEC, BREAK_GAP_SEC, LOG_DIR, MODEL_PATH, OPENAI_MODEL, SMOOTH_SEC, STEP_SEC,
                    STRESS_TURNS)
from llm import OpenAIChat, TestChat, api_key_present
from prompting import band, build_system_prompt, has_code, zone
from signals import Recording, kemocon_participants
from text_affect import TextAffect, TextTracker

HISTORY_MESSAGES = 20   # recent chat messages sent to the LLM with each request
SOURCES = {"model": "Replay → sensor model",
           "labels": "Replay → recorded self-report",
           "manual": "Manual sliders"}
SOURCE_HELP = ("Sensor model: a K-EmoCon recording is replayed in real time and the trained model "
               "estimates valence/arousal from it, as it will for a live user. Recorded self-report: "
               "the same replay, but using what that participant reported (ground truth). "
               "Manual: you set the readings yourself (Wizard of Oz).")
TREND_ARROW = {"rising": "up", "falling": "down", "steady": "off"}
SCALE = alt.Scale(domain=[1, 5])
POINT_KINDS, POINT_COLORS = ["sensors", "message", "combined"], ["#4c78a8", "#54a24b", "#e45756"]
INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
BLOCK_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

st.set_page_config(page_title="Emotion-aware assistant", page_icon="💬", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 3.5rem; padding-bottom: 6rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data and state
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_resource(show_spinner="Loading the K-EmoCon recording...")
def load_recording(pid):
    rec = Recording(pid)
    reports = rec.self_report()
    labels = {int(s): (v, a) for s, a, v in reports[["seconds", "arousal", "valence"]].itertuples(index=False)}
    return rec, labels


@st.cache_data
def replay_participants():
    return kemocon_participants()


@st.cache_resource
def text_reader():
    return TextAffect()


ss = st.session_state


def reset_affect():
    bundle = load_bundle()
    ss.estimator = AffectEstimator(bundle) if bundle else None
    ss.self_report = Smoother("recorded self-report")
    ss.clock, ss.next_t, ss.running, ss.wall = 0.0, STEP_SEC, False, time.time()


def new_session():
    ss.session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ss.messages = []
    ss.text_tracker = TextTracker()        # tone of this conversation's messages
    ss.stress_streak, ss.last_break = 0, float("-inf")
    reset_affect()


def toggle_replay():
    ss.running = not ss.running
    ss.wall = time.time()


def advance_replay(rec, labels, source, speed):
    """Feed every 5-s window whose replay time has passed into the estimator."""
    now = time.time()
    if ss.running:
        ss.clock = min(rec.duration_s, ss.clock + (now - ss.wall) * speed)
    ss.wall = now
    while ss.next_t <= ss.clock:
        t = ss.next_t
        if source == "model" and ss.estimator:
            ss.estimator.add(t, rec.features(t))
        elif source == "labels" and t in labels:
            ss.self_report.add(t, *labels[t])
        ss.next_t += STEP_SEC
    if ss.clock >= rec.duration_s:
        ss.running = False


def current_reading(source):
    if source == "manual":
        return {"source": "manual", "valence": ss.manual_valence, "arousal": ss.manual_arousal,
                "valence_trend": "steady", "arousal_trend": "steady"}
    if source == "labels":
        return ss.self_report.reading()
    return ss.estimator.reading() if ss.estimator else None


def smoothed_series(source):
    if source == "labels":
        s = ss.self_report.series()
    elif source == "model" and ss.estimator:
        s = ss.estimator.smoother.series()
    else:
        return pd.DataFrame(columns=["t", "valence", "arousal"])
    s[["valence", "arousal"]] = s[["valence", "arousal"]].rolling(SMOOTH_SEC // STEP_SEC, min_periods=1).mean()
    return s


def fmt(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def to_markdown(text):
    """OpenAI models write maths as \\( \\) or \\[ \\] and money as $1.20. Streamlit renders maths only
    between $ signs, and would read '$1.20 ... $1.00' as a formula, so escape money and convert maths."""
    text = re.sub(r"\$(?=\d)", r"\\$", text)
    text = BLOCK_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)


def stream_reply(llm, messages):
    """Shows the reply as it arrives (formatted with to_markdown) and returns the raw text."""
    placeholder, text = st.empty(), ""
    for chunk in llm.stream(messages):
        text += chunk
        placeholder.markdown(to_markdown(text) + " ▌")
    placeholder.markdown(to_markdown(text))
    return text


# ---------------------------------------------------------------- charts
def x_axis(field):
    return alt.X(f"{field}:Q", scale=SCALE, title="Valence (negative → positive)")


def y_axis(field):
    return alt.Y(f"{field}:Q", scale=SCALE, title="Arousal (calm → activated)")


def circumplex(trail, points):
    """Valence-arousal plane: quadrant guides, the recent sensor trail, and the current sensor,
    message and combined readings (points: list of dicts with valence, arousal, kind)."""
    mid = pd.DataFrame({"m": [3]})
    corners = pd.DataFrame([
        {"valence": 1.1, "arousal": 4.85, "text": "stressed", "align": "left"},
        {"valence": 4.9, "arousal": 4.85, "text": "excited", "align": "right"},
        {"valence": 1.1, "arousal": 1.15, "text": "bored", "align": "left"},
        {"valence": 4.9, "arousal": 1.15, "text": "calm", "align": "right"}])
    layers = [alt.Chart(mid).mark_rule(color="#cccccc", strokeDash=[4, 4]).encode(x=x_axis("m")),
              alt.Chart(mid).mark_rule(color="#cccccc", strokeDash=[4, 4]).encode(y=y_axis("m"))]
    for align in ("left", "right"):
        layers.append(alt.Chart(corners[corners["align"] == align])
                      .mark_text(align=align, color="#999999", fontSize=11)
                      .encode(x=x_axis("valence"), y=y_axis("arousal"), text="text"))
    if len(trail) > 1:
        layers.append(alt.Chart(trail).mark_line(point=True, opacity=0.35, color="#4c78a8")
                      .encode(x=x_axis("valence"), y=y_axis("arousal"), order="t:Q"))
    layers.append(alt.Chart(pd.DataFrame(points)).mark_circle(opacity=0.95)
                  .encode(x=x_axis("valence"), y=y_axis("arousal"),
                          color=alt.Color("kind:N", scale=alt.Scale(domain=POINT_KINDS, range=POINT_COLORS),
                                          legend=alt.Legend(orient="bottom", title=None)),
                          size=alt.Size("kind:N", scale=alt.Scale(domain=POINT_KINDS, range=[120, 120, 280]),
                                        legend=None),
                          tooltip=["kind", alt.Tooltip("valence:Q", format=".1f"), alt.Tooltip("arousal:Q", format=".1f")]))
    return alt.layer(*layers).properties(height=300)


def timeline(series):
    long = series.melt("t", ["valence", "arousal"], var_name="reading", value_name="value")
    return (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("t:Q", title=None, axis=alt.Axis(format="d", labelExpr="datum.value + ' s'")),
                    y=alt.Y("value:Q", scale=SCALE, title=None, axis=alt.Axis(values=[1, 2, 3, 4, 5])),
                    color=alt.Color("reading:N", scale=alt.Scale(domain=["valence", "arousal"],
                                                                 range=["#4c78a8", "#f58518"]),
                                    legend=alt.Legend(orient="top", title=None)))
            .properties(height=210))


def affect_panel(source, replay, speed, show, use_text):
    """Advances the replay clock and, unless hidden, draws the affect monitor. Runs as a fragment
    every second while the replay plays, so the chat is not redrawn."""
    if replay:
        advance_replay(*replay, source, speed)
    if not show:
        return
    with st.container(border=True):
        st.markdown("##### Affect monitor")
        if replay:
            rec = replay[0]
            st.progress(ss.clock / rec.duration_s, text=f"Replaying P{rec.pid}: {fmt(ss.clock)} / {fmt(rec.duration_s)}")
        if source == "model" and ss.estimator is None:
            st.warning("No trained model found. Run `python train_va_model.py` first.")
            return
        if source == "model" and ss.estimator.calibrating:
            st.info(f"Calibrating this person's baseline: {fmt(min(ss.clock, BASELINE_SEC))} / "
                    f"{fmt(BASELINE_SEC)}. The chatbot uses a neutral tone until readings start.", icon="⏳")
        sensor = current_reading(source)
        text = ss.text_tracker.current if use_text else None
        combined = fuse(sensor, text)
        if combined is None:
            st.caption("No readings yet. Press **Play** in the sidebar or send a message.")
            return
        state, strategy = zone(combined)
        st.markdown(f"Likely state: **{state}** · strategy: **{strategy}**")
        left, right = st.columns(2)
        for col, k in ((left, "valence"), (right, "arousal")):
            col.metric(f"{k.title()} (combined)", f"{combined[k]:.1f} / 5",
                       delta=f"{band(combined[k])} · {combined[f'{k}_trend']}",
                       delta_color="off", delta_arrow=TREND_ARROW[combined[f"{k}_trend"]])
        points = [{"kind": kind, "valence": r["valence"], "arousal": r["arousal"]}
                  for kind, r in (("sensors", sensor), ("message", text), ("combined", combined)) if r]
        series = smoothed_series(source)
        st.altair_chart(circumplex(series.tail(24), points), width="stretch")
        if text:
            share = combined["text_share"]
            st.caption(f"Latest message: valence {text['valence']:.1f}, arousal {text['arousal']:.1f}, "
                       f"confidence {text['confidence']:.2f}" + (" (carried over)" if text.get("carried_over") else "")
                       + f" · cue: {text['cue']} · weight in the combined reading: valence {share['valence']:.0%}, "
                       f"arousal {share['arousal']:.0%}")
        if len(series) > 1:
            st.altair_chart(timeline(series.tail(36)), width="stretch")
        st.caption(f"Sensors: {SOURCES[source]} · " + ("set by hand" if source == "manual" else "30-s averages")
                   + " on 1–5 scales")


# ---------------------------------------------------------------- chat helpers
def describe(r):
    return "none" if r is None else f"valence {r['valence']:.1f}, arousal {r['arousal']:.1f}"


def show_meta(meta):
    with st.expander("Adaptation details"):
        text = meta["message_tone"]
        st.markdown(f"**Injected:** {meta['injected']}  \n"
                    f"**Sensors:** {describe(meta['sensor_reading'])}  \n"
                    f"**Message tone:** {describe(text)}"
                    + (f" (confidence {text['confidence']:.2f}, cue: {text['cue']})" if text else "")
                    + (f" · not available: {meta['message_tone_error']}" if meta.get("message_tone_error") else "")
                    + f"  \n**Combined:** {describe(meta['combined_reading'])}")
        st.code(meta["system_prompt"], language=None, wrap_lines=True)
        st.caption(f"{meta['model']} · reply {meta['latency_s']} s · tone check {meta['message_tone_latency_s']} s")


def log_turn(record):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / f"{ss.session_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if "session_id" not in ss or "text_tracker" not in ss:   # also upgrades sessions opened before an update
    new_session()
bundle = load_bundle()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Session")
    participant_id = st.text_input("Participant ID", key="participant_id", placeholder="e.g. S01")
    condition = st.segmented_control(
        "Condition", ["adaptive", "static"], default="adaptive", required=True, key="condition",
        format_func={"adaptive": "Adaptive", "static": "Static (control)"}.get,
        help="Adaptive: the readings are injected into the system prompt. Static: the same assistant without them.")
    participant_view = st.toggle("Participant view", key="participant_view",
                                 help="Hides the affect monitor and adaptation details while a participant chats.")
    st.button("New session", on_click=new_session, width="stretch")

    st.header("Affect input")
    source = st.radio("Readings from", list(SOURCES), format_func=SOURCES.get, key="source", help=SOURCE_HELP)
    replay, speed, pid = None, 1, None
    if source == "manual":
        st.slider("Valence", 1.0, 5.0, 3.0, 0.1, key="manual_valence", help="1 = very negative, 5 = very positive")
        st.slider("Arousal", 1.0, 5.0, 3.0, 0.1, key="manual_arousal", help="1 = very calm, 5 = very activated")
    else:
        pid = st.selectbox("K-EmoCon participant", replay_participants(), format_func=lambda p: f"P{p}",
                           key="replay_pid")
        speed = st.segmented_control("Replay speed", [1, 2, 5, 10], default=1, required=True,
                                     format_func=lambda s: f"{s}×", key="speed")
        replay = load_recording(pid)
    if ss.get("affect_key") != (source, pid):
        ss.affect_key = (source, pid)
        reset_affect()
    use_text = st.toggle("Read the tone of messages", value=True, key="use_text",
                         help="Rates each message's valence/arousal with a small OpenAI call and combines it "
                              "with the sensors (valence leans on the message, arousal is split). Off = sensors only.")
    if replay:
        left, right = st.columns(2)
        left.button("Pause" if ss.running else "Play", on_click=toggle_replay, type="primary", width="stretch")
        right.button("Restart", on_click=reset_affect, width="stretch")
    if source == "model" and bundle:
        with st.expander("Model quality on unseen people"):
            for k in ("valence", "arousal"):
                m = bundle["metrics"][k]
                st.markdown(f"**{k.title()}**: average error {m['MAE']:.2f} vs "
                            f"{m['MAE (always predict the average)']:.2f} for always guessing the average; "
                            f"follows the person's ups and downs in {m['people with positive correlation']} people.")
            st.caption(f"{bundle['trained_on']}. On new people this model is not yet better than guessing "
                       "the average, so treat its readings as a placeholder until a better model is trained.")

    st.header("Language model")
    test_mode = st.toggle("Test mode (no API calls)", value=not api_key_present(), key="test_mode")
    if test_mode:
        st.caption("Replies show what would be sent to the model. Message tone is not read in test mode.")
    elif api_key_present():
        st.caption(f"OpenAI · `{OPENAI_MODEL}` · key loaded from .env")
    else:
        st.error("No OPENAI_API_KEY in .env. Add it and restart the app.")

# ---------------------------------------------------------------- main area
st.title("Emotion-aware assistant")
if not participant_view:
    st.caption(f"Session {ss.session_id} · {condition} · {SOURCES[source]}")

if participant_view:
    chat_area, panel_area = st.container(), st.empty()
else:
    chat_area, panel_area = st.columns([3, 2], gap="large")

with panel_area:
    st.fragment(affect_panel, run_every=1.0 if ss.running else None)(source, replay, speed, not participant_view,
                                                                     use_text)

with chat_area:
    messages_box = st.container(height=560, border=False, autoscroll=True)
    with messages_box:
        if not ss.messages:
            with st.chat_message("assistant"):
                st.markdown("Hi! I can help you with puzzles, code and other problem-solving tasks. "
                            "What are you working on?")
        for m in ss.messages:
            with st.chat_message(m["role"]):
                st.markdown(to_markdown(m["content"]))
                if m.get("meta") and not participant_view:
                    show_meta(m["meta"])

if user_text := st.chat_input("Message the assistant"):
    ss.messages.append({"role": "user", "content": user_text})
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-HISTORY_MESSAGES:]]

    with messages_box:
        with st.chat_message("user"):
            st.markdown(to_markdown(user_text))
        with st.chat_message("assistant"):
            # 1) Tone of the message: read and logged in both conditions, injected only in the adaptive one.
            typing = st.empty()
            typing.markdown("_…_")
            tone, tone_error, t0 = None, None, time.time()
            if use_text and not test_mode:
                try:
                    tone = text_reader().rate(history)
                except Exception as e:     # the reply still goes out, based on the sensors alone
                    tone_error = str(e)[:200]
            tone_latency = round(time.time() - t0, 2)
            typing.empty()
            tone_used = ss.text_tracker.update(tone) if use_text else None
            sensor = current_reading(source)
            combined = fuse(sensor, tone_used)
            state, strategy = zone(combined) if combined else (None, None)

            # 2) A short pause may be suggested only after sustained stress, never mid-task, at most every 10 min.
            ss.stress_streak = ss.stress_streak + 1 if strategy == "de-escalate" else 0
            allow_break = (condition == "adaptive" and ss.stress_streak >= STRESS_TURNS
                           and not has_code(user_text) and time.time() - ss.last_break >= BREAK_GAP_SEC)
            if allow_break:
                ss.last_break = time.time()
            cue = tone_used["cue"] if tone_used and not tone_used.get("carried_over") else None
            system_prompt, injected = build_system_prompt(combined, adaptive=condition == "adaptive",
                                                          cue=cue, allow_break=allow_break)

            # 3) The reply.
            started = time.time()
            try:
                llm = TestChat() if test_mode else OpenAIChat()
                reply = stream_reply(llm, [{"role": "system", "content": system_prompt}] + history)
            except Exception as e:
                ss.messages.pop()
                st.error(f"The language model failed: {e}")
                st.stop()
            meta = {"condition": condition, "affect_source": source, "sensor_reading": sensor,
                    "message_tone": tone, "message_tone_used": tone_used, "message_tone_error": tone_error,
                    "message_tone_latency_s": tone_latency, "combined_reading": combined, "state": state,
                    "strategy": strategy, "pause_allowed": allow_break, "injected": injected,
                    "system_prompt": system_prompt, "model": llm.name, "latency_s": round(time.time() - started, 2)}
            if replay:
                meta.update(replay_pid=replay[0].pid, replay_seconds=round(ss.clock))
            if not participant_view:
                show_meta(meta)

    ss.messages.append({"role": "assistant", "content": reply, "meta": meta})
    log_turn({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session_id": ss.session_id,
              "participant_id": participant_id, "turn": sum(m["role"] == "user" for m in ss.messages),
              "user": user_text, "assistant": reply, **meta})
    st.rerun()   # redraw the affect monitor with this message's tone (it was drawn before the message was read)
