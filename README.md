# Student Academic Risk Prediction

A self-contained Jupyter notebook using **scikit-learn on CPU** to estimate failure
and passing probabilities from academic and behavioral inputs. It supports mathematics
and Portuguese with separate models, each with or without previous-period grades.
The interactive form highlights the most likely outcome and the strongest individual
SHAP contribution to failure probability.

## Setup and run

Use **Python 3.12** (the tested interpreter). CUDA and PyTorch are unnecessary.
From this project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name student-risk --display-name "Student Risk (Python)"
.\.venv\Scripts\python.exe -m jupyterlab student_risk_analysis.ipynb
```

If Python 3.12 is your default, use `python -m venv .venv` for the first command.
For an existing environment, skip its creation. Activation is optional: the explicit
interpreter paths above avoid PowerShell activation-policy issues.

In JupyterLab, select **Student Risk (Python)**, then **Restart Kernel and Run All**.
Wait for training and the verification section to complete. In section 8, select a
subject and prediction mode, answer every active field, and click **Predict**.
G1/G2 appear only in the grades mode. The initial SHAP explanation may take longer
while numerical routines initialize. Subsequent predictions reuse the trained models.

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.
Start Jupyter from the repository root so that `dataset/` resolves correctly.
GitHub's notebook viewer shows saved results; the interactive form needs JupyterLab
and a live kernel. After restarting the kernel, rerun the notebook before predicting.

## Files and API

- `student_risk_analysis.ipynb`: problem definition, input codebook, exploratory
  charts, train-only model selection, held-out results, explanations, student form,
  executable verification, and limitations.
- `requirements.txt`: pinned direct dependencies used for verification.
- `scripts/verify_notebook.py`: fresh-kernel execution and output checks.
- `dataset/`: the supplied CSV files and original codebook/merge reference.
- `Student_Performance_Analysis_using_Machine_Learnin.pdf`: unchanged reference paper.

The notebook exposes:

```python
inputs = {
    "studytime": 2, "absences": 6, "failures": 0, "traveltime": 2,
    "schoolsup": "no", "famsup": "yes", "activities": "yes",
    "higher": "yes", "internet": "yes",
}
result = predict_student("math", "without_grades", inputs)
result["probabilities"]      # {"pass": ..., "fail": ...}, sums to 1
result["most_likely_outcome"]
result["prominent_factor"]
result["feature_contributions"]  # ranked by absolute contribution
show_prediction(result)

# Later in the course, with both period grades available:
result = predict_student("portuguese", "with_grades", {**inputs, "G1": 8, "G2": 9})
```

Accepted subjects are `math` and `portuguese`; modes are `without_grades` and
`with_grades`. The nine base inputs are required. G1/G2 are required only in the
grades mode. Extra fields, including G3, are rejected rather than silently ignored.
Validation uses the notebook's displayed codebook. The original failures description
is inconsistent with the files: this project uses their observed 0-3 categories.
The prediction function does not submit inputs or write student records to files.
Jupyter can preserve widget values when saving widget state; clear the form before
saving or sharing a notebook containing real student responses.

## Evaluation and interpretation

Failure means **G3 < 10/20**, passing means **G3 >= 10/20**. G3 is never an input.
Each subject has a seed-42 stratified 80/20 split shared by both modes. Five-fold
training CV selects the lowest mean log loss, with higher failure recall as the tie
breaker. Candidates are a prior baseline, logistic regression, a depth-limited tree,
and random forest; trees have sigmoid calibration within three inner training folds.
All preprocessing is inside pipelines, and the test records do not select models.

The notebook reports log loss, Brier score, failure precision/recall/F1, ROC-AUC,
PR average precision, accuracy, confusion matrices, and reliability plots. PR average
precision is the step-weighted summary, not a claim to report trapezoidal PR area.
The displayed outcome uses a fixed **50% failure threshold**; exact ties are flagged
as Fail. Both probabilities are always shown.

Local SHAP explanations evaluate the final probability-producing model using 30
training-only background records and 20 forward/reverse permutation cycles. They
operate on original input fields, decoding categorical values before prediction.
Contributions are percentage-point differences relative to that background. The
largest absolute contribution can either raise or lower risk. If a constant prior
model wins, the output reports that no input factor has a measurable contribution.

These are small, historical Portuguese secondary-school datasets. Behavioral inputs
have no timestamps, so no-grades predictions are retrospective estimates rather than
validated early warnings. SHAP factors describe model behavior, not causes. Do not
assume the model transfers to another institution without new validation. Several
students appear in both subjects; subjects are trained and evaluated separately.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\verify_notebook.py
```

The verification script starts a fresh kernel using its own interpreter, executes
every cell, and checks for successful verification output. It writes an executed
copy under ignored `artifacts/`. Use `--save` to update the main notebook's outputs
after a successful run. It tests all subject/mode combinations, invalid inputs,
class ordering, split isolation, explanation additivity and repeatability, and the
same callbacks used by the form. The saved notebook includes charts and result tables.

Measured held-out results from the documented seed-42 split:

| Subject | Input mode | Accuracy | Failure recall | ROC-AUC |
|---|---|---:|---:|---:|
| Mathematics | Without grades | 70.9% | 30.8% | 0.747 |
| Mathematics | With G1/G2 | 92.4% | 88.5% | 0.972 |
| Portuguese | Without grades | 83.1% | 10.0% | 0.735 |
| Portuguese | With G1/G2 | 94.6% | 90.0% | 0.955 |

Previous grades improve results on this split. Without them, the fixed 50% threshold
misses many failing students; high overall accuracy should not be read as strong
failure detection. Full probability-quality and baseline comparisons are in the notebook.

## Sources and attribution

- Kalpana, P., Arunmaran, E., Hanif, S., & Deebak, T. (2020).
  *Student Performance Analysis using Machine Learning*. IJITEE, 9(6), 211-215.
  [DOI: 10.35940/ijitee.F3585.049620](https://doi.org/10.35940/ijitee.F3585.049620).
  The supplied paper is included unchanged under its stated **CC BY-NC-ND** license.
  This project extends the paper's idea; it does not claim an exact reproduction.
- Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository.
  [DOI: 10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T).
  [Dataset description](https://archive.ics.uci.edu/dataset/320/student+performance).
  Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
  [probability calibration](https://scikit-learn.org/stable/modules/calibration.html),
  and [SHAP PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).

ZIP archives, local environments, and temporary artifacts are ignored by Git.
