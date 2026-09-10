"""CPU-only identity regression checks, with no framework or data dependency."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import validate


class IdentityTests(unittest.TestCase):
    def copy_model_directories(self, source, root):
        catalog=json.loads((source/'catalog.json').read_text())
        for item in catalog['architectures']:
            directory=Path(item['config_file']).parent
            shutil.copytree(source/directory, root/directory)

    def record(self):
        return {'request': {'model': {'name': 'resnet50'}, 'framework': {'api': 'V4'}, 'data': {'cohort': 'small'}, 'training': {'seed': 3416}}, 'files': {'selected.pt': 'a'*64}, 'acceptance': {'run': 'first'}}

    def test_identity_order_and_independent_dimensions(self):
        record = self.record()
        self.assertEqual(validate.artifact_id(record), validate.artifact_id(dict(reversed(list(record.items())))))
        for group, key, value in [('framework', 'api', 'V5'), ('data', 'cohort', 'full'),
                                   ('training', 'seed', 3417), ('checkpoint', 'sha256', 'b' * 64)]:
            changed = copy.deepcopy(record)
            (changed['files'] if group == 'checkpoint' else changed['request'][group])[key] = value
            self.assertNotEqual(validate.artifact_id(record), validate.artifact_id(changed))
        changed = copy.deepcopy(record)
        changed['acceptance']['run'] = 'second'
        self.assertNotEqual(validate.artifact_id(record), validate.artifact_id(changed))

    def test_catalog_and_tampering(self):
        source = Path(__file__).resolve().parent
        self.assertEqual(validate.validate(source)['architectures'], len(json.loads((source/'catalog.json').read_text())['architectures']))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(source / 'catalog.json', root)
            self.copy_model_directories(source, root)
            catalog = json.loads((root / 'catalog.json').read_text())
            config = root / catalog['architectures'][0]['config_file']
            cfg = json.loads(config.read_text())
            cfg['input_channels'] = 9
            config.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                validate.validate(root)

    def test_duplicate_names_rejected(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.copy_model_directories(source, root)
            catalog = json.loads((source / 'catalog.json').read_text())
            catalog['architectures'].append(catalog['architectures'][0])
            (root / 'catalog.json').write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'Duplicate architecture'):
                validate.validate(root)

    def test_nonfinite_configuration(self):
        with self.assertRaises(ValueError):
            validate.digest({'lr': float('nan')})


if __name__ == '__main__':
    unittest.main()
