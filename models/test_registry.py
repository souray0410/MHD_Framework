"""CPU-only identity regression checks, with no framework or data dependency."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import validate


class IdentityTests(unittest.TestCase):
    def record(self):
        return {'architecture': {'id': 'resnet50'}, 'framework': {'api': 'V4'},
                'data': {'cohort': 'small'}, 'initialization': {'kind': 'random'},
                'training': {'seed': 3416}, 'run_id': '5ac164cd-f292-4b48-8de5-c7a672c39604',
                'checkpoint': {'sha256': 'a' * 64}}

    def test_identity_order_and_independent_dimensions(self):
        record = self.record()
        self.assertEqual(validate.artifact_id(record), validate.artifact_id(dict(reversed(list(record.items())))))
        for group, key, value in [('framework', 'api', 'V5'), ('data', 'cohort', 'full'),
                                   ('training', 'seed', 3417), ('checkpoint', 'sha256', 'b' * 64)]:
            changed = copy.deepcopy(record)
            changed[group][key] = value
            self.assertNotEqual(validate.artifact_id(record), validate.artifact_id(changed))
        changed = copy.deepcopy(record)
        changed['run_id'] = '91f7e571-51b3-4e2c-b34a-fbf2e4ca977d'
        self.assertNotEqual(validate.artifact_id(record), validate.artifact_id(changed))

    def test_catalog_and_tampering(self):
        source = Path(__file__).resolve().parent
        self.assertEqual(validate.validate(source)['architectures'], 10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(source / 'catalog.json', root)
            shutil.copytree(source / 'configs', root / 'configs')
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
            shutil.copytree(source / 'configs', root / 'configs')
            catalog = json.loads((source / 'catalog.json').read_text())
            catalog['architectures'].append(catalog['architectures'][0])
            (root / 'catalog.json').write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, 'Duplicate architecture'):
                validate.validate(root)

    def test_invalid_uuid_and_nonfinite_configuration(self):
        record = self.record()
        record['run_id'] = 'not-a-run-id'
        with self.assertRaises(ValueError):
            validate.artifact_id(record)
        with self.assertRaises(ValueError):
            validate.digest({'lr': float('nan')})


if __name__ == '__main__':
    unittest.main()
