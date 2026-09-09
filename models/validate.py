"""Validate public architecture metadata and complete-bundle manifest identities."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def artifact_id(record):
    payload=dict(record);payload.pop('artifact_id',None)
    return 'model_'+digest(payload)


def validate(root):
    root=Path(root).resolve()
    catalog=json.loads((root/'catalog.json').read_text())
    names=set()
    for item in catalog['architectures']:
        key=item['architecture_id']
        if key in names: raise ValueError('Duplicate architecture ID: '+key)
        names.add(key)
        readme=(root/item['readme_file']).resolve()
        if not readme.is_relative_to(root) or not readme.is_file():
            raise ValueError('Missing model README inside models directory: '+key)
        path=(root/item['config_file']).resolve()
        if not path.is_relative_to(root): raise ValueError('Configuration outside models directory')
        config=json.loads(path.read_text())
        if config['architecture_id']!=key or digest(config)!=item['config_sha256']:
            raise ValueError('Architecture configuration identity mismatch: '+key)
    identities=set()
    for record in catalog['trained_artifacts']:
        if record['format']!='mhd_model_bundle_v1' or record['artifact_id']!=artifact_id(record):
            raise ValueError('Trained bundle identity mismatch')
        if record['request_id']!=digest(record['request']): raise ValueError('Training request identity mismatch')
        if record['artifact_id'] in identities: raise ValueError('Duplicate bundle')
        identities.add(record['artifact_id'])
    return {'architectures':len(names),'trained_artifacts':len(identities),
            'identity_checks':'passed','weights_loaded':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    print(json.dumps(validate(p.parse_args().root),indent=2))
