from __future__ import annotations

import pytest

import speaker_identity_plan0071_d0 as d0


def _exposure() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "test",
        "status": "development_only",
    }
    for key, (count, hash_key) in d0.EXPOSURE_COLLECTIONS.items():
        items = [f"{key}-{index}" for index in range(count)]
        value[key] = items
        value[hash_key] = d0._hash(items)
    return d0._content(value)


def test_validate_exposure_binds_all_collection_lengths_and_hashes() -> None:
    result = d0._validate_exposure(_exposure())

    assert set(result) == set(d0.EXPOSURE_COLLECTIONS)
    assert result["full_recordings"]["count"] == 12
    assert result["decision_rows"]["count"] == 39


def test_validate_exposure_rejects_integer_in_place_of_collection() -> None:
    exposure = _exposure()
    exposure["full_recordings"] = 12
    exposure = d0._content(exposure)

    with pytest.raises(d0.Plan0071D0Error, match="12-item list"):
        d0._validate_exposure(exposure)


def test_validate_exposure_rejects_inherited_set_hash_drift() -> None:
    exposure = _exposure()
    exposure["decision_row_set_sha256"] = "0" * 64
    exposure = d0._content(exposure)

    with pytest.raises(d0.Plan0071D0Error, match="decision_rows hash drifted"):
        d0._validate_exposure(exposure)
