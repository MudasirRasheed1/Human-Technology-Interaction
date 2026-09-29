# Evaluation protocol (all approaches)

The chatbot will always meet **new people**. An approach is only useful if it works on people it has never seen. Published papers often report 90%+ accuracy on K-EmoCon. When tested on new people, the same papers drop to ~65–75% on a 2-class task. That's no better than always predicting the most common class (see `literature_review.md`). These rules exist so our numbers mean something.

## Rules

1. **Split by person, never by window.** No person appears in both training and test data. If people interacted in pairs or groups (e.g. K-EmoCon debate partners), keep the whole group on one side.
2. **Select models with cross-validation inside the training people** (leave-one-person-out or leave-one-group-out). Touch the test people **once**, at the end.
3. **Always report baselines** next to your model:
   - *majority class*: always predict the most common label;
   - *prior only* (if you use calibration): predict from the person's calibration labels alone, with no sensors.

   If you can't beat these, say so.
4. **Report metrics that can't be gamed by class imbalance:** macro-F1 and balanced accuracy, in addition to accuracy.
5. **Report the confidence/abstention trade-off:** accuracy on answered windows vs. coverage (the share of windows answered). Choose thresholds on cross-validation, then check them on test.
6. **Only use past data at each moment.** Any smoothing or context window must use only samples before the prediction time. Never smooth or modify the ground-truth labels.
7. **State the label source and class definition,** e.g. "self-reports, 3 classes: 1–2 / 3 / 4–5". Say where "neutral" goes.
8. **Report the spread across people** (std or per-person results), not just the mean.

## Reporting template

| Item | Your value |
|---|---|
| Dataset, number of people (train / test) | |
| Labels (source, classes, class balance) | |
| Split method | |
| Model and inputs | |
| Test accuracy / macro-F1 / balanced accuracy | |
| Majority-class baseline on the same test set | |
| Accuracy at 25% / 50% / 100% coverage | |
