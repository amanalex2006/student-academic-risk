"""Integration checks against the notebook's exported models; run unittest discover."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from app import create_app
from risk_app.storage import MODEL_KEYS, MODEL_IDS

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {'studytime': 2, 'absences': 6, 'failures': 0, 'traveltime': 2,
           'schoolsup': 'no', 'famsup': 'yes', 'activities': 'yes',
           'higher': 'yes', 'internet': 'yes'}


class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / 'models' / 'manifest.json').exists():
            raise RuntimeError('Export models first: python scripts/train_models.py')
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_pages_and_catalog(self):
        for path in ['/', '/models', '/static/app.js', '/static/styles.css', '/static/favicon.svg']:
            with self.client.get(path) as response:
                self.assertEqual(response.status_code, 200)
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        models = response.json['models']
        self.assertEqual({m['id'] for m in models}, set(MODEL_IDS))
        for model in models:
            self.assertGreater(model['size_bytes'], 0)
            self.assertEqual(len(model['sha256']), 64)
            self.assertEqual(model['filename'], model['id'] + '.joblib')
            self.assertIn('failure_recall', model['metrics'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_all_four_saved_models_without_training(self):
        with patch.object(Pipeline, 'fit', side_effect=AssertionError('Prediction tried to train')), \
             patch.object(CalibratedClassifierCV, 'fit', side_effect=AssertionError('Prediction tried to calibrate')):
            for subject, mode in MODEL_KEYS:
                profile = {**PROFILE, **({'G1': 8, 'G2': 9} if mode == 'with_grades' else {})}
                with self.subTest(subject=subject, mode=mode):
                    response = self.client.post('/api/predict', json={'subject': subject, 'mode': mode, 'inputs': profile})
                    self.assertEqual(response.status_code, 200, response.json)
                    result = response.json
                    self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
                    self.assertEqual(result['most_likely_outcome'], 'Fail' if result['probabilities']['fail'] >= .5 else 'Pass')
                    self.assertLess(abs(result['explanation_residual']), 1e-6)
                    self.assertEqual(len(result['feature_contributions']), 11 if mode == 'with_grades' else 9)
                    self.assertEqual(result['prominent_factor'], result['feature_contributions'][0])
                    self.assertNotIn('G3', [c['feature'] for c in result['feature_contributions']])

    def test_invalid_requests_return_readable_errors(self):
        cases = [
            {'subject': 'math', 'mode': 'without_grades', 'inputs': {}},
            {'subject': 'unknown', 'mode': 'without_grades', 'inputs': PROFILE},
            {'subject': 'math', 'mode': 'with_grades', 'inputs': PROFILE},
            {'subject': 'math', 'mode': 'without_grades', 'inputs': {**PROFILE, 'G3': 10}},
            {'subject': 'math', 'mode': 'without_grades', 'inputs': {**PROFILE, 'absences': -1}},
            {'subject': 'math', 'mode': 'without_grades', 'inputs': {**PROFILE, 'internet': 'maybe'}},
            {'subject': 'math', 'mode': 'without_grades', 'inputs': {**PROFILE, 'absences': 2.5}},
            {'subject': 'math', 'mode': [], 'inputs': PROFILE},
            [], None,
        ]
        for payload in cases:
            response = self.client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
            self.assertEqual(response.status_code, 400, response.json)
            self.assertTrue(response.json['error'])
        self.assertEqual(self.client.post('/api/predict', data='hello').status_code, 415)
        self.assertEqual(self.client.post('/api/predict', data='{', content_type='application/json').status_code, 400)

    def test_missing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory).test_client()
            self.assertEqual(client.get('/api/models').json['models'], [])
            result = client.post('/api/predict', json={'subject': 'math', 'mode': 'without_grades', 'inputs': PROFILE})
            self.assertEqual(result.status_code, 503)
            self.assertIn('No trained models', result.json['error'])

    def test_corrupt_artifact_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (ROOT / 'models').glob('*'):
                if path.is_file():
                    shutil.copy2(path, directory)
            (Path(directory) / 'math-without_grades.joblib').write_bytes(b'corrupt model')
            client = create_app(directory).test_client()
            response = client.post('/api/predict', json={'subject': 'math', 'mode': 'without_grades', 'inputs': PROFILE})
            self.assertEqual(response.status_code, 503)
            self.assertIn('integrity', response.json['error'])

    def test_invalid_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (ROOT / 'models').glob('*'):
                if path.is_file():
                    shutil.copy2(path, directory)
            manifest_path = Path(directory) / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['models'][0]['filename'] = '../../untrusted.joblib'
            manifest_path.write_text(json.dumps(manifest))
            response = create_app(directory).test_client().get('/api/models')
            self.assertEqual(response.status_code, 503)

    def test_dependency_mismatch_has_a_recovery_message(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (ROOT / 'models').glob('*'):
                if path.is_file():
                    shutil.copy2(path, directory)
            manifest_path = Path(directory) / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['models'][0]['packages']['scikit-learn'] = '0.0.0'
            manifest_path.write_text(json.dumps(manifest))
            client = create_app(directory).test_client()
            response = client.post('/api/predict', json={'subject': 'math', 'mode': 'without_grades', 'inputs': PROFILE})
            self.assertEqual(response.status_code, 503)
            self.assertIn('dependency versions', response.json['error'])


if __name__ == '__main__':
    unittest.main()
