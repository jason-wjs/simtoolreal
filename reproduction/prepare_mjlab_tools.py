"""CPU-only preparation of the original 1200-tool pool for the mjlab port."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mjlabenvs.assets import audit_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Import the original pure asset utilities without invoking Isaac Lab's
    # environment-registration package. No source copy or alternate generator.
    source = ROOT / 'isaacsimenvs/tasks/simtoolreal/utils'
    package = types.ModuleType('_original_simtoolreal_assets')
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(package.__name__ + '.generate_objects', source / 'generate_objects.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    paths, scales = generator.generate_handle_head_urdfs(
        ('hammer', 'screwdriver', 'marker', 'spatula', 'eraser', 'brush'),
        num_per_type=100, out_dir=output / 'urdf', seed=42, object_base_size=0.04)
    assert len(paths) == 1200
    rows = []
    for path, scale in zip(paths, scales):
        path = Path(path)
        rows.append({'path': str(path.relative_to(output)), 'scale': list(scale),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **audit_object(path)})
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'generator_hashes': {n: hashlib.sha256((source/n).read_bytes()).hexdigest()
                             for n in ['generate_objects.py', 'object_size_distributions.py']},
        'tools': rows, 'scope': 'Original Lab procedural geometry/mass/COM/inertia and shuffled pool order. CPU MuJoCo compilation only; robot/control/contact/training port not yet accepted.'}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'tools': len(rows), 'all_passed': all(r['passed'] for r in rows),
                      'max_inertia_error': max(r['max_inertia_error'] for r in rows)}))


if __name__ == '__main__':
    main()
