"""Use persisted models and training-only backgrounds; no training on requests."""
from threading import RLock
import numpy as np
import shap

from .schema import (
    SUBJECT_LABELS, MODE_LABELS, FEATURE_OPTIONS, FEATURE_LABELS,
    validate_inputs, probability_map, encode_for_explanation,
    decode_for_prediction, direction_for, FAILURE_THRESHOLD,
)
from .storage import ModelStorageError


class Predictor:
    def __init__(self, store):
        self.store = store
        self.explainers = {}
        # SHAP uses NumPy's global random state. Serialize explanations to keep
        # repeatable results under a multithreaded local web server.
        self.lock = RLock()

    def predict(self, subject, mode, inputs):
        if not isinstance(subject, str) or subject not in SUBJECT_LABELS:
            raise ValueError('Choose Mathematics or Portuguese.')
        frame = validate_inputs(mode, inputs)
        identifier = f'{subject}-{mode}'
        with self.lock:
            artifact = self.store.load(identifier)
            metadata = artifact['metadata']
            features = metadata['features']
            if list(frame.columns) != features:
                raise ModelStorageError('Saved model inputs differ from the website. Re-export the notebook.')
            estimator = artifact['estimator']
            probabilities = {k: float(v[0]) for k, v in probability_map(estimator, frame).items()}
            rng_state = np.random.get_state()
            try:
                np.random.seed(42)
                if identifier not in self.explainers:
                    def model_function(matrix):
                        return probability_map(estimator, decode_for_prediction(matrix, features))['fail']
                    masker = shap.maskers.Independent(
                        encode_for_explanation(artifact['background']), max_samples=30)
                    self.explainers[identifier] = shap.PermutationExplainer(
                        model_function, masker, feature_names=features, seed=42)
                np.random.seed(42)
                explanation = self.explainers[identifier](
                    encode_for_explanation(frame), max_evals=20 * (2 * len(features) + 1), silent=True)
            finally:
                np.random.set_state(rng_state)
            values = np.asarray(explanation.values[0], dtype=float).reshape(-1)
            base = float(np.asarray(explanation.base_values[0]).reshape(-1)[0])
            residual = base + float(values.sum()) - probabilities['fail']
            if abs(residual) > 1e-6:
                raise RuntimeError('Explanation did not reconcile with predicted risk.')
            contributions = []
            for feature, value in zip(features, values):
                raw = frame.iloc[0][feature]
                if isinstance(raw, np.generic):
                    raw = raw.item()
                contributions.append({
                    'feature': feature, 'label': FEATURE_LABELS[feature], 'value': raw,
                    'contribution': float(value), 'percentage_points': float(100 * value),
                    'direction': direction_for(value),
                })
            contributions.sort(key=lambda c: -abs(c['contribution']))
            prominent = contributions[0] if abs(contributions[0]['contribution']) >= 1e-10 else None
            notes = []
            for name, (lo, hi) in artifact['training_ranges'].items():
                if not lo <= inputs[name] <= hi:
                    notes.append(f'{FEATURE_LABELS[name]} is outside the model training range ({lo}-{hi}); this estimate has less data support.')
            return {
                'subject': subject, 'mode': mode, 'model': metadata['model_name'],
                'model_id': identifier, 'probabilities': probabilities,
                'most_likely_outcome': 'Fail' if probabilities['fail'] >= FAILURE_THRESHOLD else 'Pass',
                'failure_threshold': FAILURE_THRESHOLD,
                'is_probability_tie': bool(np.isclose(probabilities['fail'], .5, atol=1e-12, rtol=0)),
                'base_failure_probability': base, 'feature_contributions': contributions,
                'prominent_factor': prominent, 'explanation_residual': residual,
                'notes': notes,
            }
