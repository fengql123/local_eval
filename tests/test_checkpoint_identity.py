"""Two HF layouts and two acquisition identity boundaries."""

import pytest

from eval.models import checkpoint_revision


@pytest.mark.parametrize(
    "files",
    [
        {"config.json": "commit-a", "model.safetensors": "commit-a"},
        {
            "Q4_K_M/model-00001-of-00002.gguf": "commit-a",
            "Q4_K_M/model-00002-of-00002.gguf": "commit-a",
        },
    ],
)
def test_records_actual_acquired_commit_without_requiring_config(tmp_path, files):
    for name, commit in files.items():
        path = tmp_path / ".cache/huggingface/download" / (name + ".metadata")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(commit + "\netag\n123.0\n")
    assert checkpoint_revision(tmp_path) == "commit-a"


def test_non_hf_store_has_no_hf_revision(tmp_path):
    assert checkpoint_revision(tmp_path) is None


def test_mixed_download_revisions_cannot_authorize_one_quality_profile(tmp_path):
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for number, revision in enumerate(("commit-a", "commit-b")):
        (metadata / f"part-{number}.gguf.metadata").write_text(revision + "\netag\n123.0\n")
    with pytest.raises(ValueError, match="different HF revisions"):
        checkpoint_revision(tmp_path)
