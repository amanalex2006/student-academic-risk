"""Run the local website: python app.py (http://127.0.0.1:8000)."""
from pathlib import Path
import argparse
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from risk_app.prediction import Predictor, SUBJECT_LABELS, MODE_LABELS, FEATURE_OPTIONS, FEATURE_LABELS
from risk_app.storage import ModelStore, ModelStorageError

ROOT = Path(__file__).resolve().parent


def create_app(model_directory=None):
    app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    store = ModelStore(model_directory or ROOT / 'models')
    predictor = Predictor(store)
    app.extensions['model_store'] = store
    app.extensions['predictor'] = predictor

    @app.get('/')
    @app.get('/models')
    def index():
        return render_template('index.html', page='models' if request.path == '/models' else 'assessment')

    @app.get('/api/models')
    def models():
        try:
            return jsonify({'models': store.catalog(), 'storage_directory': 'models/'})
        except ModelStorageError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.get('/api/config')
    def config():
        return jsonify({'subjects': SUBJECT_LABELS, 'modes': MODE_LABELS,
                        'options': FEATURE_OPTIONS, 'labels': FEATURE_LABELS})

    @app.post('/api/predict')
    def predict():
        try:
            if not request.is_json:
                return jsonify({'error': 'Send student inputs as JSON.'}), 415
            payload = request.get_json()
            if not isinstance(payload, dict) or set(payload) != {'subject', 'mode', 'inputs'}:
                raise ValueError('Provide subject, mode, and inputs only.')
            return jsonify(predictor.predict(payload['subject'], payload['mode'], payload['inputs']))
        except (ValueError, BadRequest) as exc:
            return jsonify({'error': str(exc)}), 400
        except ModelStorageError as exc:
            return jsonify({'error': str(exc)}), 503
        except Exception:
            app.logger.exception('Student prediction failed')
            return jsonify({'error': 'Prediction could not be completed. Check the local server and saved models.'}), 500

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app


if __name__ == '__main__':
    from waitress import serve
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'Student Risk website: http://127.0.0.1:{args.port}', flush=True)
    serve(create_app(), host='127.0.0.1', port=args.port, threads=4)
