from pathlib import Path

import acoustic_generation5_review_repair as repair


def _card(reference: str, label: str = "A") -> dict:
    return {
        "ordinal": 1,
        "speaker_label": label,
        "speaker_ref": reference,
        "clip": {"snippets": [{"text": "A useful transcript clue."}]},
    }


def test_standalone_review_embeds_audio_prefills_and_valid_javascript(tmp_path):
    cards = [_card(f"Candidate {index} / Speaker A") for index in range(1, 41)]
    preview = {"private_evidence": {"cards": cards}}
    for card in cards:
        (tmp_path / f"01-a.wav").write_bytes(b"R" * 100)
    page = repair.render_standalone_review(
        preview,
        tmp_path,
        answers={"Candidate 1 / Speaker A": "Chris Williams"},
    )
    assert page.count("data:audio/wav;base64,") == 40
    assert 'value="Chris Williams"' in page
    assert "lines.join('\\n')" in page
    assert "lines.join('\n')" not in page
    assert "<details open>" in page


def test_standalone_review_rejects_unknown_prefill_reference(tmp_path):
    preview = {"private_evidence": {"cards": [_card(f"Candidate {index} / Speaker A") for index in range(1, 41)]}}
    (tmp_path / "01-a.wav").write_bytes(b"R" * 100)
    try:
        repair.render_standalone_review(preview, tmp_path, answers={"not a card": "Person"})
    except repair.Generation5ReviewRepairError as exc:
        assert "reference is unknown" in str(exc)
    else:
        raise AssertionError("unknown prefill reference was accepted")
