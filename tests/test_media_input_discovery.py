from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcribe_common import expand_audio_inputs


def test_existing_bracketed_recording_is_a_literal_path(tmp_path):
    source = tmp_path / 'meeting [raw recording].mp4'
    source.write_bytes(b'fixture')
    assert expand_audio_inputs([str(source)]) == [source.resolve()]


def test_literal_path_does_not_select_a_different_glob_match(tmp_path):
    source = tmp_path / '[ab].wav'
    decoy = tmp_path / 'a.wav'
    source.write_bytes(b'requested recording')
    decoy.write_bytes(b'other recording')
    assert expand_audio_inputs([str(source)]) == [source.resolve()]


def test_bracketed_parent_directory_and_home_expansion(tmp_path, monkeypatch):
    folder = tmp_path / 'recordings [2026]'
    folder.mkdir()
    source = folder / 'meeting.wav'
    source.write_bytes(b'fixture')
    monkeypatch.setenv('HOME', str(tmp_path))
    assert expand_audio_inputs(['~/recordings [2026]/meeting.wav']) == [
        source.resolve()
    ]


def test_patterns_expand_and_duplicate_paths_are_removed(tmp_path):
    sources = [tmp_path / 'a.wav', tmp_path / 'b.wav']
    for source in sources:
        source.write_bytes(b'fixture')
    nested = tmp_path / 'nested'
    nested.mkdir()
    deeper = nested / 'c.wav'
    deeper.write_bytes(b'fixture')
    actual = expand_audio_inputs([
        str(tmp_path / '[ab].wav'), str(tmp_path / '**/*.wav'), str(sources[0]),
    ])
    assert set(actual) == {p.resolve() for p in [*sources, deeper]}
    assert len(actual) == 3


def test_unmatched_pattern_and_directory_are_skipped(tmp_path, capsys):
    assert expand_audio_inputs([str(tmp_path / 'missing*.wav'), str(tmp_path)]) == []
    output = capsys.readouterr()
    assert 'no files matched' in output.err
    assert 'Skipping non-file match' in output.err
