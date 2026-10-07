import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'manage_global_memory_guidance.py'
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / 'templates' / 'global-graphiti-guidance.md'
)


def load_module():
    spec = importlib.util.spec_from_file_location('manage_global_memory_guidance', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GlobalMemoryGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_template_preserves_exact_disposition_contract(self):
        text = TEMPLATE_PATH.read_text(encoding='utf-8')

        self.assertIn('exactly one explicit memory disposition', text)
        for disposition in (
            'queued',
            'duplicate_noop',
            'not_durable',
            'forbidden',
            'unavailable',
        ):
            self.assertIn(f'`{disposition}`', text)
        self.assertIn('graphiti-runtime remember', text)
        self.assertIn('graphiti-runtime remember-reconcile', text)
        self.assertNotIn('If yes, write', text)
        self.assertIn('canonical source artifact is expected provenance', text)
        self.assertIn('not by itself a reason to select `not_durable`', text)
        self.assertIn('semantic qualification separate from effect authority', text)
        self.assertIn('select `forbidden`', text)
        self.assertIn('unreceipted memory job is an observability failure', text)

    def test_shared_memory_module_preserves_curation_calibration(self):
        source = Path(__file__).resolve().parents[2] / 'modules' / 'graph-backed-memory-usage.md'
        bundled = (
            Path(__file__).resolve().parents[1]
            / 'policy-library'
            / 'modules'
            / 'graph-backed-memory-usage.md'
        )
        text = source.read_text(encoding='utf-8')

        self.assertEqual(source.read_bytes(), bundled.read_bytes())
        self.assertIn('canonical source artifact is expected provenance', text)
        self.assertIn('not by itself a reason to select `not_durable`', text)
        self.assertIn('semantic qualification separate from effect authority', text)
        self.assertIn('unreceipted direct job is an observability failure', text)

    def test_install_replaces_only_the_managed_section_and_is_idempotent(self):
        original = textwrap.dedent(
            '''
            # Global Guidance

            Keep this prefix.

            ## Graphiti Discovery And Memory

            - At closeout, decide whether memory is useful. If yes, write it.

            ## Optional repository MCP servers

            Keep this suffix.
            '''
        ).lstrip()

        first = self.module.install_guidance_text(original, TEMPLATE_PATH.read_text())
        second = self.module.install_guidance_text(first, TEMPLATE_PATH.read_text())

        self.assertEqual(first, second)
        self.assertIn('Keep this prefix.', first)
        self.assertIn('Keep this suffix.', first)
        self.assertIn('exactly one explicit memory disposition', first)
        self.assertNotIn('decide whether memory is useful', first)

    def test_audit_separates_policy_observation_from_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / 'receipts'
            receipts.mkdir()
            (receipts / 'queued-remember.json').write_text(
                json.dumps(
                    {
                        'schema_version': 'closeout_memory_receipt.v0',
                        'recorded_at': '2026-10-01T10:00:00+00:00',
                        'disposition': 'queued',
                        'status': 'queued',
                        'job_id': 'job-receipted',
                        'graphiti_writes_attempted': 1,
                    }
                ),
                encoding='utf-8',
            )
            (receipts / 'nonwrite-remember.json').write_text(
                json.dumps(
                    {
                        'schema_version': 'closeout_memory_receipt.v0',
                        'recorded_at': '2026-10-01T11:00:00+00:00',
                        'disposition': 'not_durable',
                        'status': 'recorded',
                        'reason': 'canonical issue is authoritative',
                        'job_id': None,
                        'graphiti_writes_attempted': 0,
                    }
                ),
                encoding='utf-8',
            )
            jobs = root / 'jobs.json'
            jobs.write_text(
                json.dumps(
                    {
                        'jobs': [
                            {
                                'job_id': 'job-receipted',
                                'queued_at': '2026-10-01T10:00:01+00:00',
                                'status': 'failed',
                            },
                            {
                                'job_id': 'job-direct',
                                'queued_at': '2026-10-01T12:00:00+00:00',
                                'status': 'completed',
                            },
                        ]
                    }
                ),
                encoding='utf-8',
            )

            report = self.module.audit_closeouts(
                receipts_dir=receipts,
                job_store=jobs,
                since='2026-10-01T00:00:00+00:00',
                expected_closeouts=3,
            )

        self.assertEqual(report['observed_disposition_count'], 2)
        self.assertEqual(report['missing_disposition_count'], 1)
        self.assertEqual(report['disposition_counts'], {'not_durable': 1, 'queued': 1})
        self.assertEqual(report['queued_delivery_status_counts'], {'failed': 1})
        self.assertEqual(report['unreceipted_memory_job_ids'], ['job-direct'])


if __name__ == '__main__':
    unittest.main()
