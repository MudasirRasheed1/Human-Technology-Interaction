# Emotion-aware chatbot (valence/arousal version)

```
wearable signals (every 5 s)                         user message
  -> features of the last 10 s       signals.py         -> tone: valence, arousal, confidence, cue
  -> z-scores vs first 2 min         affect_model.py       (small OpenAI call)            text_affect.py
  -> valence + arousal model (1-5)   models/va_model.joblib
  -> 30-s average + trend            affect_model.py
                 \                                       /
                  combined reading (valence leans on the message, arousal is split)   affect_model.fuse
                  -> readings + calming rubric (+ pause permission) in the system prompt   prompting.py
                  -> OpenAI chat model -> reply                                            llm.py, app.py
                  -> everything logged to logs/<session>.jsonl
```

## Run it
1. Put your key in `.env` (`OPENAI_API_KEY=...`). Optional lines: `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT` (reasoning models only: `low` is the default; `none` is faster but less careful), `OPENAI_TEMPERATURE` (non-reasoning models only), `TEXT_AFFECT_MODEL` (model that rates message tone; defaults to `OPENAI_MODEL`).
2. In a terminal in this folder:
   ```
   conda activate torch
   python train_va_model.py     # only needed again if you change the features or the model
   streamlit run app.py
   ```
Without a key the app starts in **Test mode** (replies show what would have been sent; message tone is not read).

## How the bot adapts
- **Message tone.** Every message is rated for valence/arousal (1-5), with a confidence (how much emotion the message reveals) and a short cue. Messages like "ok" or a plain question reveal little, so the previous estimate carries over and fades (`TEXT_DECAY`).
- **Combining.** Sensors mostly show activation, words show positive/negative, so valence takes up to 80% from the message and arousal 50% (`TEXT_WEIGHT`, scaled by confidence). Big disagreements are pointed out to the model.
- **Calming goal.** The rubric tells the model to steer the user toward feeling calm, capable and engaged: de-escalate when stressed, ground when tense, re-engage when discouraged or bored, keep momentum when excited, stay steady when calm, and return gradually to a normal style once readings recover. It never mentions sensors or emotions.
- **Pause suggestion.** Allowed only after `STRESS_TURNS` (2) stressed messages in a row, never when the message contains code or an error, and at most once every 10 minutes.
- **Accuracy first.** Both conditions are told to check the user's answers before judging them; adapting tone must never make an answer less correct.

All weights and thresholds are in `config.py`.

## Using the app (sidebar)
- **Readings from**: *Replay → sensor model* (a K-EmoCon recording streamed in real time; the first 2 minutes calibrate), *Replay → recorded self-report* (that person's own ratings), or *Manual sliders* (Wizard of Oz).
- **Read the tone of messages**: on by default; off = sensors only.
- **Condition**: *Adaptive* injects the readings; *Static (control)* gets none. Message tone is still read and logged in the static condition, so you can compare how users' tone changes in both.
- **Participant view** hides the monitor and adaptation details; also collapse the sidebar before a participant sits down.
- **Participant ID** is saved with every logged turn.

## Logged per turn
participant, condition, sensor reading, message tone (new and the one used), combined reading, likely state and strategy, whether a pause was allowed, the full system prompt, reply, model, tone-check and reply latency.

## Model quality
`train_va_model.py` tests the sensor model on people it never saw. It is currently **not better than always guessing the average rating** (see the sidebar), so its readings stay near the middle and the message tone does most of the work. Retrain with better features or another model and the app picks up the new `models/va_model.joblib`.

## Files
| File | What it does |
|---|---|
| `config.py` | paths, timing, combining weights, loads `.env` |
| `signals.py` | E4 wristband features per 10-s window; K-EmoCon loading |
| `affect_model.py` | personal baseline, sensor valence/arousal, smoothing, combining with message tone |
| `text_affect.py` | rates each message's tone; carry-over of informative estimates |
| `train_va_model.py` | builds the training set from raw K-EmoCon, evaluates, saves the model |
| `prompting.py` | system prompt: readings, calming rubric, pause permission |
| `llm.py` | OpenAI streaming client (handles reasoning and non-reasoning models) and the offline test mode |
| `app.py` | Streamlit UI, replay engine, logging |

`.env` and `logs/` are in `.gitignore`, so the key and participant data are never committed.
