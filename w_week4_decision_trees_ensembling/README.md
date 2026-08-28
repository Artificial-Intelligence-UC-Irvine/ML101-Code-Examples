# Winter Week 4 — Decision Trees & Ensembling

This folder contains the coding activity for the AI@UCI Winter Week 4 workshop.

## Files
- `fill-in-blanks.ipynb` — student version used during the live workshop
- `master.ipynb` — completed instructor / answer-key version
- `requirements.txt` — Python dependencies

## Workshop flow
The notebook mirrors the slide narrative:

`data → try splits → measure impurity → choose a split → grow a tree → control depth → combine trees → test`

Students:
1. visualize a nonlinear classification problem,
2. compute Gini impurity,
3. train and inspect a shallow decision tree,
4. compare shallow and deep trees,
5. connect tree depth to overfitting,
6. visualize decision regions,
7. train a random forest,
8. inspect individual tree votes,
9. compare one tree with many trees.

## Teaching structure
This notebook follows the AI@UCI workshop notebook guidance:

- Keep the same conceptual theme as the slides.
- Provide both a student fill-in-blanks notebook and a completed answer key.
- Fill the student notebook live during the workshop.
- Do not make every cell a TODO.
- Show a completed example before asking students to implement a similar step.
- Keep plotting/helper code completed when plotting syntax is not the learning objective.

## Suggested live flow
1. Run the nonlinear feature-space visualization.
2. Complete the train/test split.
3. Show the completed 4-vs-1 Gini example.
4. Have students calculate Gini for a 3-vs-3 group.
5. Train the completed depth-2 tree.
6. Inspect the tree rules and diagram.
7. Have students train a deeper tree and compare accuracy.
8. Run the depth-vs-accuracy plot.
9. Compare shallow and deep decision regions.
10. Have students build the random forest.
11. Show individual tree votes.
12. Compare deep-tree and random-forest test accuracy.

## Timing
Core activity: about 15–20 minutes.

If short on time, keep:
- Gini impurity
- one decision tree
- `max_depth`
- train/test comparison
- random forest
- final one-tree vs. many-trees comparison

The tree diagram, full depth curve, and boosting discussion can be shortened or skipped.

## Environment
Python 3 with NumPy, pandas, matplotlib, and scikit-learn.
