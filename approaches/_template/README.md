> **This is an empty template, not a real approach.** Copy this folder to `approaches/<dataset>_<method>/`, create the folders listed under *Layout*, and replace this README. For a finished example, see [`../kemocon_physio_ml/`](../kemocon_physio_ml/).

# <Approach name>

**Owner:** <name> · **Dataset:** <dataset> · **Inputs:** <sensors / voice / text> · **Status:** <in progress / done>

## Idea
One paragraph: what signal, which model or method, and why it could work.

## Layout
```
src/          code (data preparation, features, training, evaluation)
notebooks/    one walkthrough notebook explaining what was done, how and why
docs/         dataset description, design decisions
reports/      results (RESULTS.md, figures/, metrics json/csv)
data/         NOT committed; see data/README.md for how to get the data
requirements.txt
```

## Results
Fill in the reporting template from `../../docs/evaluation_protocol.md`.

## Reproduce
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```
