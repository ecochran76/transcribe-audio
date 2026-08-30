from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from conversation_evidence_adapters import AdapterSourceScope
from conversation_evidence_mail_receipts import (
    MailReceiptsAdapterConfig,
    MailReceiptsEvidenceAdapter,
    MailReceiptsPage,
)
from conversation_identity_retrieval import ProviderRetrievalRequest
from conversation_knowledge_evidence import EvidenceScope
from mail_evidence_normalization import NormalizedMailEvidence, normalize_mail_evidence
from relationship_role_discovery import discover_mail_relationship_hypotheses


FIXTURE = (
    Path(__file__).parents[1]
    / "docs/dev/fixtures/plan-0073-p0/discovery-scenarios.json"
)
PAGE_SCOPE = {
    "source_profile_id": "mail-receipts-default",
    "account_id": "account-redacted",
    "tenant_id": "tenant-redacted",
    "namespace": "namespace-redacted",
    "corpus_id": "corpus-redacted",
}


@dataclass
class FixtureReader:
    first_address: str
    records: tuple[Mapping[str, Any], ...]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def service_profile(self) -> dict[str, Any]:
        return {
            "profile": "operator-lite",
            "capabilities": ["search_mail"],
            "mailbox_mutation": False,
            "corpus_operation_execution": False,
        }

    def search_exact_email(self, **kwargs: Any) -> MailReceiptsPage:
        self.calls.append(dict(kwargs))
        return MailReceiptsPage(
            records=(
                self.records if kwargs["address"] == self.first_address else ()
            ),
            as_of="2026-01-07T16:00:00Z",
            source_scope=PAGE_SCOPE,
        )


def test_frozen_plan0073_scenarios_replay_through_p1_p2_and_p3() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contacts = {
        value["contact_id"]: {"email": address, **value}
        for address, value in fixture["contacts"].items()
    }
    contacts["contact-redacted-account"] = {
        "contact_id": "contact-redacted-account",
        "label": "Account Contact",
        "email": fixture["account_address"],
        "contact_class": "person_candidate",
    }

    for case in fixture["cases"]:
        addresses = sorted(
            {
                address
                for record in case["records"]
                for address in (*record["from"], *record["to"], *record["cc"])
                if address != fixture["account_address"]
            }
        )
        records = []
        for raw in case["records"]:
            participants = (*raw["from"], *raw["to"], *raw["cc"])
            records.append(
                {
                    **raw,
                    "contact_ids_by_address": {
                        address: next(
                            contact_id
                            for contact_id, contact in contacts.items()
                            if contact["email"] == address
                        )
                        for address in participants
                    },
                }
            )
        reader = FixtureReader(addresses[0], tuple(records))
        adapter = MailReceiptsEvidenceAdapter(
            config=MailReceiptsAdapterConfig(
                scope=AdapterSourceScope(
                    source_profile_id="mail-receipts-default",
                    provider_kind="mail_receipts",
                    account_id="account-redacted",
                    tenant_id="tenant-redacted",
                    capabilities=("mail_metadata_read",),
                ),
                namespace="namespace-redacted",
                corpus_id="corpus-redacted",
                account_address=fixture["account_address"],
            ),
            reader=reader,
            retrieved_at="2026-01-07T16:01:00Z",
        )
        retrieved = adapter.retrieve(
            ProviderRetrievalRequest(
                conversation_id=f"fixture-{case['case_id']}",
                query_terms=tuple(addresses),
                scopes=(
                    EvidenceScope(
                        source_profile_id="mail-receipts-default",
                        account_id="account-redacted",
                        tenant_id="tenant-redacted",
                    ),
                ),
                capabilities=("mail_metadata_read",),
                as_of=fixture["as_of"],
                max_records=25,
                max_characters=1,
            )
        )
        normalized = normalize_mail_evidence(
            retrieved.snapshots,
            query_receipt=retrieved.query_receipt,
        )
        first = discover_mail_relationship_hypotheses(
            normalized.observations,
            normalized.independence_groups,
            contacts=contacts,
            account_address=fixture["account_address"],
            input_watermark=normalized.input_watermark,
        )
        reversed_projection = NormalizedMailEvidence(
            observations=tuple(reversed(normalized.observations)),
            independence_groups=tuple(reversed(normalized.independence_groups)),
            input_watermark=normalized.input_watermark,
        )
        second = discover_mail_relationship_hypotheses(
            reversed_projection.observations,
            reversed_projection.independence_groups,
            contacts=contacts,
            account_address=fixture["account_address"],
            input_watermark=reversed_projection.input_watermark,
        )

        assert first == second, case["case_id"]
        expected = case["expected"]
        if "hypothesis_kinds" in expected:
            assert {
                item["hypothesis_kind"] for item in first.hypotheses
            } == set(expected["hypothesis_kinds"]), case["case_id"]
        if "independence_group_count" in expected:
            assert len(normalized.independence_groups) == expected[
                "independence_group_count"
            ]
        if "duplicate_count" in expected:
            assert sum(
                group["duplicate_count"]
                for group in normalized.independence_groups
            ) == expected["duplicate_count"]
        if "independent_thread_count" in expected and first.hypotheses:
            assert max(
                item["independent_thread_count"] for item in first.hypotheses
            ) == expected["independent_thread_count"]
        if "role_values" in expected:
            assert sorted(
                item["counterpart_label"]
                for item in first.hypotheses
                if item["hypothesis_kind"] == "contextual_role"
            ) == expected["role_values"]
        if "conflict_count" in expected:
            assert sum(
                len(item["conflicts"]) for item in first.hypotheses
            ) == expected["conflict_count"]
        actual_reasons = set(first.excluded_reason_counts)
        actual_reasons.update(
            group["reason_code"]
            for group in normalized.independence_groups
            if group["reason_code"]
        )
        assert actual_reasons == set(expected["excluded_reason_codes"]), case[
            "case_id"
        ]
