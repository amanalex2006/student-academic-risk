"""Execute the notebook in a fresh project-environment kernel and check outputs."""
from pathlib import Path
import argparse
import tempfile
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true', help='Save the successfully executed notebook in place.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / 'student_risk_analysis.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # A temporary kernel spec guarantees that verification uses this interpreter,
    # even if another global python3 or student-risk kernel was registered earlier.
    with tempfile.TemporaryDirectory(prefix='student-risk-kernel-') as directory:
        import json
        kernel_dir = Path(directory) / 'verification'
        kernel_dir.mkdir()
        (kernel_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Student risk verification', 'language': 'python',
        }), encoding='utf-8')
        manager = KernelSpecManager(kernel_dirs=[directory])
        from jupyter_client import KernelManager
        km = KernelManager(kernel_name='verification', kernel_spec_manager=manager)
        def started(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                print(f'Executing cell {cell_index + 1}/{len(notebook.cells)}: '
                      f'{cell.source.splitlines()[0][:90]}', flush=True)
        client = NotebookClient(notebook, km=km, kernel_name='verification', timeout=900,
                                resources={'metadata': {'path': str(root)}},
                                allow_errors=False, store_widget_state=True,
                                on_cell_start=started)
        try:
            executed = client.execute()
        except Exception:
            failure = root / 'artifacts' / 'failed_execution.ipynb'
            failure.parent.mkdir(exist_ok=True)
            nbformat.write(notebook, failure)
            print(f'Failed execution preserved at {failure}', file=sys.stderr)
            raise
    errors = [o for c in executed.cells if c.cell_type == 'code'
              for o in c.get('outputs', []) if o.output_type == 'error']
    assert not errors
    assert all(c.execution_count is not None for c in executed.cells if c.cell_type == 'code')
    assert any('widget callbacks passed.' in o.get('text', '')
               for c in executed.cells if c.cell_type == 'code' for o in c.get('outputs', []))
    nbformat.validate(executed)
    if args.save:
        nbformat.write(executed, path)
        print('Saved executed notebook with analysis outputs and widget state.', flush=True)
    else:
        output = root / 'artifacts' / 'verified_notebook.ipynb'
        output.parent.mkdir(exist_ok=True)
        nbformat.write(executed, output)
        print(f'Verified notebook saved to {output}', flush=True)
    print('Fresh-kernel verification passed.', flush=True)


if __name__ == '__main__':
    main()
