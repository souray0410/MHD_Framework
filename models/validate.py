"""Validate catalog identities and configuration digests; does not load weights."""
import argparse
import hashlib
import json
from pathlib import Path
import uuid


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def training_config_id(record):
    return digest({key: record[key] for key in
                   ('architecture', 'framework', 'data', 'initialization', 'training')})


def artifact_id(record):
    return 'weights_' + digest({'training_config_id': training_config_id(record),
                                'run_id': str(uuid.UUID(record['run_id'])),
                                'checkpoint_sha256': record['checkpoint']['sha256']})


def validate(root):
    root = Path(root).resolve()
    catalog = json.loads((root / 'catalog.json').read_text())
    names = set()
    for item in catalog['architectures']:
        key = item['architecture_id']
        if key in names:
            raise ValueError('Duplicate architecture ID: ' + key)
        names.add(key)
        path = (root / item['config_file']).resolve()
        if root not in path.parents:
            raise ValueError('Configuration outside models directory')
        config = json.loads(path.read_text())
        if config['architecture_id'] != key or digest(config) != item['config_sha256']:
            raise ValueError('Architecture configuration identity mismatch: ' + key)
    weights = set()
    for record in catalog['trained_artifacts']:
        expected = artifact_id(record)
        if record['artifact_id'] != expected or record['training_config_id'] != training_config_id(record):
            raise ValueError('Trained artifact identity mismatch')
        if expected in weights:
            raise ValueError('Duplicate trained artifact ID: ' + expected)
        if record['architecture']['id'] not in names:
            raise ValueError('Unregistered architecture')
        weights.add(expected)
    return {'architectures': len(names), 'trained_artifacts': len(weights),
            'identity_checks': 'passed', 'weights_loaded': False,
            'scope': 'identity and configuration checks; not scientific acceptance'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(validate(parser.parse_args().root), indent=2))
