"""Prediction helpers matching the validated notebook's original input contract."""
from collections.abc import Mapping
from numbers import Real
import numpy as np
import pandas as pd


SEED = 42

FAILURE_THRESHOLD = 0.50

BASE_FEATURES = [
    'studytime', 'absences', 'failures', 'traveltime', 'schoolsup',
    'famsup', 'activities', 'higher', 'internet',
]

GRADE_FEATURES = ['G1', 'G2']

SUBJECT_LABELS = {'math': 'Mathematics', 'portuguese': 'Portuguese'}

MODE_LABELS = {
    'without_grades': 'Without previous grades',
    'with_grades': 'With G1 and G2',
}

FEATURE_LABELS = {
    'studytime': 'Weekly study time', 'absences': 'School absences',
    'failures': 'Past class failures', 'traveltime': 'Travel time to school',
    'schoolsup': 'Extra school support', 'famsup': 'Family educational support',
    'activities': 'Extracurricular activities', 'higher': 'Plans higher education',
    'internet': 'Internet access at home', 'G1': 'First-period grade',
    'G2': 'Second-period grade',
}

FEATURE_OPTIONS = {
    'studytime': [('<2 hours/week', 1), ('2-5 hours/week', 2),
                  ('5-10 hours/week', 3), ('>10 hours/week', 4)],
    'traveltime': [('<15 minutes', 1), ('15-30 minutes', 2),
                   ('30-60 minutes', 3), ('>60 minutes', 4)],
    'failures': [('0', 0), ('1', 1), ('2', 2), ('3', 3)],
    **{name: [('No', 'no'), ('Yes', 'yes')] for name in
       ['schoolsup', 'famsup', 'activities', 'higher', 'internet']},
}

INTEGER_RANGES = {'absences': (0, 93), 'G1': (0, 20), 'G2': (0, 20)}

CATEGORICAL_FEATURES = list(FEATURE_OPTIONS)

def features_for(mode):
    if not isinstance(mode, str) or mode not in MODE_LABELS:
        raise ValueError(f"mode must be one of {list(MODE_LABELS)}; received {mode!r}.")
    return BASE_FEATURES + (GRADE_FEATURES if mode == 'with_grades' else [])

def validate_inputs(mode, inputs):
    """Return a one-row DataFrame in model order; never silently fill user inputs."""
    features = features_for(mode)
    if not isinstance(inputs, Mapping):
        raise ValueError('inputs must be a dictionary mapping field names to values.')
    missing = [name for name in features if name not in inputs]
    extra = [name for name in inputs if name not in features]
    if missing:
        raise ValueError('Missing required inputs: ' + ', '.join(missing))
    if extra:
        raise ValueError('Unexpected inputs for this mode: ' + ', '.join(map(str, extra)))
    clean = {}
    for name in features:
        value = inputs[name]
        if name in FEATURE_OPTIONS:
            allowed = [raw for _, raw in FEATURE_OPTIONS[name]]
            if isinstance(allowed[0], str):
                if not isinstance(value, str) or value not in allowed:
                    raise ValueError(f'{name} must be one of {allowed}.')
                clean[name] = value
                continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or float(value) != int(value)):
            raise ValueError(f'{name} must be a finite integer.')
        value = int(value)
        if name in FEATURE_OPTIONS:
            if value not in allowed:
                raise ValueError(f'{name} must be one of {allowed}.')
        else:
            lo, hi = INTEGER_RANGES[name]
            if not lo <= value <= hi:
                raise ValueError(f'{name} must be between {lo} and {hi}, inclusive.')
        clean[name] = value
    return pd.DataFrame([clean], columns=features)

def probability_map(estimator, frame):
    classes = list(estimator.classes_)
    assert set(classes) == {0, 1}, 'Model must know both outcome classes.'
    raw = np.asarray(estimator.predict_proba(frame), dtype=float)
    assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    assert np.allclose(raw.sum(axis=1), 1.0, atol=1e-10)
    return {'pass': raw[:, classes.index(0)], 'fail': raw[:, classes.index(1)]}

def encode_for_explanation(frame):
    encoded = frame.copy()
    for column in encoded.columns:
        if column in FEATURE_OPTIONS and isinstance(FEATURE_OPTIONS[column][0][1], str):
            encoded[column] = encoded[column].map({'no': 0, 'yes': 1})
    return encoded.to_numpy(dtype=float)

def decode_for_prediction(matrix, features):
    frame = pd.DataFrame(np.asarray(matrix, dtype=float), columns=features)
    for column in features:
        if column in FEATURE_OPTIONS and isinstance(FEATURE_OPTIONS[column][0][1], str):
            if not frame[column].isin([0, 1]).all():
                raise ValueError(f'Explanation generated an invalid category for {column}.')
            frame[column] = frame[column].map({0: 'no', 1: 'yes'})
        else:
            if not np.isfinite(frame[column]).all() or not (frame[column] == np.floor(frame[column])).all():
                raise ValueError(f'Explanation generated a noninteger value for {column}.')
            frame[column] = frame[column].astype(int)
            if column in FEATURE_OPTIONS:
                allowed = [raw for _, raw in FEATURE_OPTIONS[column]]
                if not frame[column].isin(allowed).all():
                    raise ValueError(f'Explanation generated an invalid category for {column}.')
            else:
                lo, hi = INTEGER_RANGES[column]
                if not frame[column].between(lo, hi).all():
                    raise ValueError(f'Explanation generated an out-of-range value for {column}.')
    return frame

def direction_for(value):
    if abs(value) < 1e-10:
        return 'no measurable contribution'
    return 'raises failure risk' if value > 0 else 'lowers failure risk'
