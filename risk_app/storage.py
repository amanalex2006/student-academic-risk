"""Export fitted notebook pipelines and inspect a fixed local model registry."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import uuid

import joblib

MODEL_KEYS = tuple((s, m) for s in ('math', 'portuguese')
                   for m in ('without_grades', 'with_grades'))
MODEL_IDS = tuple(f'{s}-{m}' for s, m in MODEL_KEYS)
SCHEMA_VERSION = 1
COMPATIBILITY_PACKAGES = ('scikit-learn', 'numpy', 'pandas', 'scipy')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_registry(registry, datasets, test_results, directory):
    """Serialize already-trained pipelines; never fit or change their test results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if set(registry) != set(MODEL_KEYS):
        raise ValueError('Export requires all four subject/mode combinations.')
    packages = {p: version(p) for p in (*COMPATIBILITY_PACKAGES, 'shap', 'joblib')}
    exported_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for subject, mode in MODEL_KEYS:
        item = registry[(subject, mode)]
        identifier = f'{subject}-{mode}'
        training = datasets[subject].loc[item['train_ids'], item['features']]
        selected = test_results[(test_results.subject == subject)
                                & (test_results['mode'] == mode)
                                & (test_results.role == 'selected')].iloc[0]
        baseline = test_results[(test_results.subject == subject)
                                & (test_results['mode'] == mode)
                                & (test_results.role == 'baseline')].iloc[0]
        metric_names = ['log_loss', 'brier_score', 'failure_precision', 'failure_recall',
                        'failure_f1', 'roc_auc', 'pr_average_precision', 'accuracy']
        metadata = {
            'id': identifier, 'subject': subject, 'mode': mode,
            'model_name': item['model_name'], 'features': list(item['features']),
            'trained_at': exported_at, 'train_records': len(item['train_ids']),
            'test_records': len(item['test_ids']), 'failure_threshold': .5,
            'cv_log_loss': float(item['selection']['cv_log_loss']),
            'metrics': {k: float(selected[k]) for k in metric_names},
            'baseline_metrics': {k: float(baseline[k]) for k in metric_names},
            'packages': packages,
            'dataset_sha256': hashlib.sha256(datasets[subject].to_csv(index=False).encode()).hexdigest(),
        }
        artifact = {
            'schema_version': SCHEMA_VERSION, 'metadata': metadata,
            'estimator': item['estimator'],
            'background': training.sample(n=min(30, len(training)), random_state=42),
            'training_ranges': {k: [int(training[k].min()), int(training[k].max())]
                                for k in ['absences', 'G1', 'G2'] if k in training},
        }
        path = directory / f'{identifier}.joblib'
        temporary = directory / f'{identifier}.{uuid.uuid4().hex}.tmp'
        try:
            joblib.dump(artifact, temporary, compress=3)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        entries.append({**metadata, 'filename': path.name,
                        'size_bytes': path.stat().st_size, 'sha256': checksum(path)})
    manifest = {'schema_version': SCHEMA_VERSION, 'exported_at': exported_at, 'models': entries}
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(temporary, directory / 'manifest.json')
    return manifest


class ModelStorageError(RuntimeError):
    pass


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}

    def catalog(self):
        manifest_path = self.directory / 'manifest.json'
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            models = manifest['models']
            if manifest['schema_version'] != SCHEMA_VERSION:
                raise ValueError('Unsupported model manifest version.')
            if len(models) != 4 or {m['id'] for m in models} != set(MODEL_IDS):
                raise ValueError('Model registry must contain the four expected models.')
            for entry in models:
                if entry['filename'] != f"{entry['id']}.joblib":
                    raise ValueError('Unexpected model filename.')
                if not (self.directory / entry['filename']).is_file():
                    raise ValueError(f"Saved model is missing: {entry['id']}.")
            return models
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ModelStorageError('The saved model registry is incomplete or unreadable. Re-export the training notebook.') from exc

    def load(self, identifier):
        if identifier not in MODEL_IDS:
            raise ValueError('Unknown model identifier.')
        entries = {m['id']: m for m in self.catalog()}
        if identifier not in entries:
            raise ModelStorageError('No trained models are saved yet. Run the training notebook and save its models.')
        entry = entries[identifier]
        if identifier in self.cache:
            if self.cache[identifier]['checksum'] != entry['sha256']:
                raise ModelStorageError('Saved models changed. Restart the website to load the new models.')
            return self.cache[identifier]['artifact']
        for package in COMPATIBILITY_PACKAGES:
            if entry['packages'].get(package) != version(package):
                raise ModelStorageError('Saved models use different dependency versions. Use the training environment or re-export the notebook.')
        path = self.directory / entry['filename']
        if checksum(path) != entry['sha256']:
            raise ModelStorageError('A model file failed its integrity check. Re-export the training notebook.')
        try:
            artifact = joblib.load(path)
            if artifact['schema_version'] != SCHEMA_VERSION or artifact['metadata']['id'] != identifier:
                raise ValueError('Model identity mismatch.')
        except Exception as exc:
            raise ModelStorageError('A saved model could not be loaded. Re-export the training notebook.') from exc
        self.cache[identifier] = {'artifact': artifact, 'checksum': entry['sha256']}
        return artifact
