# Fall Week 8 — Linear Classification

This folder contains the coding activity for the AI@UCI Week 8 workshop.

## Files
- `fill-in-blanks.ipynb` — student version used during the live workshop
- `master.ipynb` — completed answer key / instructor version
- `requirements.txt` — Python dependencies

## Workshop flow
The notebook mirrors the slide narrative:

`feature space → score → prediction → mistake → update → repeat → test`

Students build a small binary linear classifier from scratch using a perceptron-style update, visualize the learned decision boundary, evaluate on unseen data, and finally compare the implementation with `sklearn.linear_model.Perceptron`.

## Teaching structure
This notebook follows the AI@UCI workshop guidance:

- Keep the same conceptual theme as the slides.
- Use a student fill-in-blanks notebook and a completed answer key.
- Fill the student notebook live while presenting.
- Do not make every cell a TODO.
- Show a completed example first, then ask students to implement a similar step.
- Keep plotting/helper code completed when syntax is not the teaching goal.

## Suggested live flow
1. Run the feature-space visualization.
2. Complete the train/test split.
3. Show the completed score example.
4. Have students compute a second score and prediction.
5. Implement `predict_one`.
6. Show one completed perceptron update.
7. Have students repeat the update on a second example.
8. Complete the training loop.
9. Compare the boundary before vs. after training.
10. Compute train/test accuracy.
11. Run the sklearn reference model.

## Timing
Core activity: about 15–20 minutes.

If short on time, keep:
- score → prediction
- one manual update
- training loop
- learned boundary
- test accuracy

The sklearn comparison and final extension can be shortened or skipped.

## Environment
Python 3 with NumPy, pandas, matplotlib, and scikit-learn.
