"""Finite deterministic training grids; no dataset or scheduler assumptions."""
import copy
import itertools

from .artifacts import digest, resolve_or_request, validate_request


def expand_grid(base, grid):
    """Expand explicit dotted fields, rejecting misspellings and duplicates.

    Every candidate remains a separate model/data/training identity. Selection
    policy belongs to the approved study; this function never inspects metrics.
    """
    keys = sorted(grid)
    for key in keys:
        if not grid[key]:
            raise ValueError('Empty grid axis: '+key)
    seen = set()
    for values in itertools.product(*(grid[k] for k in keys)):
        request = copy.deepcopy(base)
        for key,value in zip(keys,values):
            path = key.split('.')
            target = request
            for segment in path[:-1]:
                target=target[segment]
            if path[-1] not in target:
                raise ValueError('Unknown grid field: '+key)
            target[path[-1]]=value
        validate_request(request)
        identity = digest(request)
        if identity not in seen:
            seen.add(identity)
            yield request


def prepare_grid(store, base, grid):
    return [resolve_or_request(store, r) for r in expand_grid(base, grid)]


def ensure_trained(store, request, *, trainer=None):
    """Resolve or execute one explicitly supplied native-training callback.

    The callback receives (request, attempt_directory) and must return an accepted
    complete bundle *inside this store*. It owns the approved data, convergence,
    resource limits and resume policy. Concurrent workers serialize this request;
    failed attempts remain recorded and no alternative hyperparameters are tried.
    Without a callback, this only registers pending work.
    """
    import fcntl
    import json
    from pathlib import Path
    import uuid
    from .artifacts import atomic_json, publish_bundle
    state = resolve_or_request(store, request)
    if state['status'] == 'ready' or trainer is None:
        return state
    directory = Path(store)/'requests'/state['request_id']
    with (directory/'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = resolve_or_request(store, request)
        if state['status'] == 'ready':
            return state
        attempt = directory/'attempts'/str(uuid.uuid4())
        attempt.mkdir(parents=True)
        atomic_json(attempt/'status.json', {'status':'running','request_id':state['request_id']})
        try:
            bundle = trainer(request, attempt)
            publish_bundle(store, bundle, request)
        except Exception as error:
            atomic_json(attempt/'status.json', {'status':'failed','error_type':type(error).__name__,
                                               'request_id':state['request_id']})
            raise
        atomic_json(attempt/'status.json', {'status':'accepted','request_id':state['request_id']})
        return resolve_or_request(store, request)
