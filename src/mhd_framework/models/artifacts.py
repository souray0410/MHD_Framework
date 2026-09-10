"""Portable full-model bundles and exact-match, deferred training requests.

Weights remain in a caller-owned store. No network access, floating 'latest',
project methods, datasets or scheduler policies are embedded here.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_request(request):
    for key in ('model', 'framework', 'data', 'initialization', 'training'):
        if not isinstance(request.get(key), dict) or not request[key]:
            raise ValueError('Missing explicit request section: '+key)
    for key in ('split_sha256', 'preprocessing_sha256', 'label_schema_sha256'):
        value = request['data'].get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing data identity: '+key)
    for field in ('preprocessing', 'label_schema'):
        if not isinstance(request['data'].get(field), dict) or digest(request['data'][field]) != request['data'][field+'_sha256']:
            raise ValueError('Complete, hash-matched '+field+' metadata is required')
    if request['framework'].get('api') not in ('V4', 'V5'):
        raise ValueError('An explicit framework API is required')
    commit = request['framework'].get('commit', '')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('An exact framework commit is required')
    if 'seed' not in request['training']:
        raise ValueError('An explicit training seed is required')
    digest(request)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def export_bundle(model, destination, request, *, acceptance, resume_state):
    """Export a selected full task model plus independent continuation state.

    resume_state must describe the STOPPING state, not optimizer tensors from
    an unrelated epoch paired with the selected model. No pickled model class
    is stored. acceptance is the caller's explicit training/replay receipt.
    """
    validate_request(request)
    if model.configuration() != request['model']:
        raise ValueError('Model configuration and request do not match')
    if acceptance.get('status') != 'accepted' or not all(acceptance.get(k) for k in
            ('training_receipt_sha256', 'replay_receipt_sha256')):
        raise ValueError('Training and replay acceptance receipts are required')
    required = ('model', 'optimizer', 'scheduler', 'epoch', 'rng', 'sampler')
    if any(k not in resume_state for k in required):
        raise ValueError('Continuation state must contain model, optimizer, scheduler, epoch, RNG and sampler')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite a model bundle')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.bundle-', dir=destination.parent))
    try:
        torch.save({'format': 'mhd_complete_model_v1', 'config': model.configuration(),
                    'state_dict': model.state_dict()}, temporary/'selected.pt')
        torch.save(resume_state, temporary/'resume.pt')
        manifest = {'format': 'mhd_model_bundle_v1', 'request': request,
            'request_id': digest(request), 'acceptance': acceptance,
            'files': {p.name: file_sha256(p) for p in temporary.iterdir()},
            'endpoints': model.endpoint_nodes, 'feature_channels': model.feature_channels}
        manifest['artifact_id'] = 'model_' + digest(manifest)
        atomic_json(temporary/'manifest.json', manifest)
        # Caller must own the destination; never overwrite an existing bundle.
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_bundle(path, expected_request=None):
    path = Path(path)
    m = json.loads((path/'manifest.json').read_text())
    identity = dict(m); artifact_id = identity.pop('artifact_id')
    if m['format'] != 'mhd_model_bundle_v1' or artifact_id != 'model_'+digest(identity):
        raise ValueError('Bundle manifest identity mismatch')
    validate_request(m['request'])
    if m['request_id'] != digest(m['request']):
        raise ValueError('Request identity mismatch')
    if expected_request is not None and digest(expected_request) != m['request_id']:
        raise ValueError('Weights do not match the requested model/data/protocol')
    if m['acceptance'].get('status') != 'accepted':
        raise ValueError('Unaccepted bundle')
    if set(m['files']) != {'selected.pt', 'resume.pt'}:
        raise ValueError('Bundle must contain complete selected and continuation checkpoints')
    for name, sha in m['files'].items():
        if file_sha256(path/name) != sha:
            raise ValueError('Checkpoint checksum mismatch: '+name)
    return m


def load_bundle(path, expected_request, *, device='cpu'):
    from mhd_framework import __api_version__
    from . import create_model
    m = verify_bundle(path, expected_request)
    if m['request']['framework']['api'] != __api_version__:
        raise ValueError('Installed framework API does not match the bundle')
    verify_runtime(m['request']['framework'])
    payload = torch.load(Path(path)/'selected.pt', map_location='cpu', weights_only=True)
    if payload['format'] != 'mhd_complete_model_v1' or payload['config'] != m['request']['model']:
        raise ValueError('Checkpoint configuration mismatch')
    with torch.random.fork_rng(devices=[]):
        model = create_model(payload['config'])
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device), m


def materialize_bundle(source, destination, expected_request):
    """Make a verified independent project copy, never a mutable symlink."""
    m = verify_bundle(source, expected_request)
    destination = Path(destination)
    if destination.exists():
        old = verify_bundle(destination, expected_request)
        if old['artifact_id'] != m['artifact_id']:
            raise FileExistsError('Project copy already refers to a different accepted run')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.copy-', dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        verify_bundle(temporary, expected_request)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def resolve_or_request(store, request):
    """Return an accepted bundle or atomically queue its exact training request.

    Does not return random weights on a miss. An external approved runner owns
    training, retries and publication; imports and inference never launch jobs.
    """
    import fcntl
    validate_request(request)
    root = Path(store)
    request_id = digest(request)
    directory = root/'requests'/request_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        accepted = directory/'accepted.json'
        if accepted.exists():
            ref = json.loads(accepted.read_text())
            bundle = (root/ref['bundle']).resolve()
            if not bundle.is_relative_to(root.resolve()):
                raise ValueError('Bundle reference escapes the model store')
            m = verify_bundle(bundle, request)
            if m['artifact_id'] != ref['artifact_id']:
                raise ValueError('Published artifact identity mismatch')
            return {'status': 'ready', 'request_id': request_id, 'bundle': str(bundle)}
        task = directory/'request.json'
        if not task.exists():
            atomic_json(task, {'status': 'pending', 'request_id': request_id, 'request': request})
        elif json.loads(task.read_text())['request'] != request:
            raise ValueError('Existing training request was changed')
        return {'status': 'pending', 'request_id': request_id, 'request_file': str(task)}


def publish_bundle(store, path, request):
    """Pin one accepted run to a request; alternatives cannot silently replace it."""
    import fcntl
    root, path = Path(store).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Published bundle must be inside the store')
    m = verify_bundle(path, request)
    resolve_or_request(root, request)
    directory = root/'requests'/digest(request)
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = directory/'accepted.json'
        ref = {'artifact_id': m['artifact_id'], 'bundle': str(path.relative_to(root))}
        if target.exists() and json.loads(target.read_text()) != ref:
            raise FileExistsError('An accepted run is already pinned; do not select by cache order')
        atomic_json(target, ref)
    return ref


def runtime_source_sha256():
    package = Path(__file__).resolve().parents[1]
    return {name: file_sha256(package/name) for name in
            ('__init__.py', 'core.py', 'utils.py', 'models/resnet.py', 'models/graph.py', 'models/densenet.py', 'models/retfound.py', 'models/vision_transformer.py', 'models/artifacts.py', 'models/recipes.py', 'models/training.py', 'models/__init__.py')}


def verify_runtime(framework):
    if framework.get('source_sha256') != runtime_source_sha256():
        raise ValueError('Installed framework/model implementation differs from the pinned bundle')
