# Fall Week 3 — Intro to ML & Basic Classifiers

This folder contains the coding activity for the AI@UCI Week 3 workshop.

## Files
- `fill-in-blanks.ipynb` — student version used during the live workshop
- `master.ipynb` — completed answer key / instructor version
- `requirements.txt` — Python dependencies

## Workshop flow
The notebook mirrors the slide narrative:

`data → features + labels → feature space → train/test split → KNN → prediction → evaluation`

The core activity uses a small synthetic Cat vs Dog dataset so the code stays focused on ML concepts rather than data-cleaning overhead.

## Teaching notes
- Run the completed visualization cells live.
- Pause at each TODO before typing the answer.
- The `k=5` sklearn example is intentionally completed first.
- Students then repeat the same pattern for `k=1` and `k=25`.
- Emphasize the gap between training and test accuracy to motivate generalization / overfitting.
- The decision-region visualization is optional if time remains.

## Environment
Python 3 with NumPy, pandas, matplotlib, and scikit-learn.
