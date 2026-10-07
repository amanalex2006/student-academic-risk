# Website verification — 7 October 2026

**Story:** a student enters a profile in the local website; the JSON API loads the
matching saved scikit-learn pipeline, computes probabilities and SHAP contributions,
and renders them alongside the most likely outcome. The model library reads the
same artifact manifest and displays its measured evaluation results.

| Boundary | Result | Evidence |
|---|---|---|
| Notebook → stored models | Passed | Four exported joblib files; reloaded probabilities and feature contributions match notebook predictions for all subject/mode combinations. |
| Website renders | Passed | Assessment and library routes return 200; desktop and 390px mobile screenshots reviewed; mobile page width equals viewport width. |
| Form → API | Passed | Mathematics without grades and Portuguese with grades submitted from the real browser form; response rendered successfully. |
| API → saved pipeline | Passed | All four combinations tested while pipeline fitting and calibration fitting were forbidden; no retraining occurred. |
| Saved pipeline → response | Passed | Probabilities sum to one; SHAP baseline plus contributions reconciles with failure probability; original feature names and correct grade mode retained. |
| Response → interface | Passed | Browser shows likelihoods, the expected highlighted factor, and its direction. Switching modes reveals G1/G2 and clears stale estimates. |
| Library → evaluation | Passed | Four cards show filenames and scores; expanded details include eight metrics compared with the prior baseline. |
| Expected failure paths | Passed | Invalid JSON/input, missing artifacts, checksum failure, unsafe manifest paths, and incompatible dependency versions tested with readable errors. |

Seven integration tests passed with `python -m unittest discover -s tests -v`.
Fresh-kernel notebook execution, dependency checks, JavaScript syntax checks, and
Git whitespace checks passed. Browser console reported zero errors or warnings.

Optional WebMCP support is feature-detected. The test browser had no registered
WebMCP tools, so browser-tool execution was not verified; the visible form and
HTTP API were verified independently.

The website remains local at `http://127.0.0.1:8000`. Generated models and the
manifest stay under ignored `models/`; ZIP files remain ignored. No student input
history or training controls are added to the website.
