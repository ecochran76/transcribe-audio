from __future__ import annotations

import json
from pathlib import Path

import pytest

import review_queue_maintenance as maintenance


def write_route_review(path: Path, route_path: Path, *, label: str = "SoyLei Tempo Chemical matter") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at": "2026-05-16T21:00:00Z",
                "reason": "Route confidence below threshold.",
                "route_decision_path": str(route_path),
                "selected_label": label,
            }
        ),
        encoding="utf-8",
    )


def test_stale_route_review_plan_only_includes_missing_routes(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    review_dir = state_root / "review-queue"
    existing_route = tmp_path / "meeting.route.json"
    existing_route.write_text("{}", encoding="utf-8")
    write_route_review(review_dir / "active.route-review.json", existing_route)
    write_route_review(review_dir / "stale.route-review.json", tmp_path / "missing.route.json")

    plan = maintenance.stale_route_review_plan(
        state_root=state_root,
        archive_root=tmp_path / "archive",
        run_id="test-run",
    )

    assert plan["summary"]["candidate_count"] == 1
    assert plan["items"][0]["id"] == "stale"
    assert plan["items"][0]["archive_path"].endswith("/test-run/stale.route-review.json")


def test_apply_stale_archive_plan_dry_run_does_not_move(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    stale_review = state_root / "review-queue" / "stale.route-review.json"
    write_route_review(stale_review, tmp_path / "missing.route.json")
    plan = maintenance.stale_route_review_plan(state_root=state_root, archive_root=tmp_path / "archive", run_id="dry")

    audit = maintenance.apply_stale_archive_plan(plan, apply=False)

    assert audit["summary"]["by_status"] == {"planned": 1}
    assert stale_review.exists()
    assert not Path(audit["results"][0]["archive_path"]).exists()


def test_apply_stale_archive_plan_moves_files(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    stale_review = state_root / "review-queue" / "stale.route-review.json"
    write_route_review(stale_review, tmp_path / "missing.route.json")
    plan = maintenance.stale_route_review_plan(state_root=state_root, archive_root=tmp_path / "archive", run_id="apply")

    audit = maintenance.apply_stale_archive_plan(plan, apply=True)

    archive_path = Path(audit["results"][0]["archive_path"])
    assert audit["summary"]["by_status"] == {"archived": 1}
    assert not stale_review.exists()
    assert archive_path.exists()


def test_run_requires_approval_token_for_apply(tmp_path: Path) -> None:
    args = maintenance.parse_args(["--state-dir", str(tmp_path / "state"), "--apply"])

    with pytest.raises(ValueError, match="approval-token"):
        maintenance.run(args)


def test_run_writes_audit_output(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    stale_review = state_root / "review-queue" / "stale.route-review.json"
    audit_path = tmp_path / "audit.json"
    write_route_review(stale_review, tmp_path / "missing.route.json")
    args = maintenance.parse_args(
        [
            "--state-dir",
            str(state_root),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--audit-output",
            str(audit_path),
        ]
    )

    audit = maintenance.run(args)

    assert audit["audit_path"] == str(audit_path.resolve())
    written = json.loads(audit_path.read_text(encoding="utf-8"))
    assert written["summary"]["by_status"] == {"planned": 1}
