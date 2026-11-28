import pytest
import torch

from causalq.models.dinov2_wrapper import convert_hpi_state
from tools.check_scs_dual_backbone import file_sha, resolve_d2_weights
from tools.convert_dinov2_for_rein import convert_state_dict


def test_hpi_and_rein_convert_the_same_tensors(monkeypatch):
    monkeypatch.setattr("tools.convert_dinov2_for_rein.inspect_state_dict", lambda _: None)
    generator = torch.Generator().manual_seed(0)
    source = {
        "patch_embed.proj.weight": torch.randn(1024, 3, 14, 14, generator=generator),
        "pos_embed": torch.randn(1, 1370, 1024, generator=generator),
        "other": torch.ones(1),
    }
    hpi = convert_hpi_state(source)
    rein = convert_state_dict(source)
    assert rein.keys() == hpi.keys()
    for key in rein:
        torch.testing.assert_close(rein[key], hpi[key], rtol=0, atol=0)


def test_resolve_prefers_a_pinned_converted_file_and_records_wrong_hash(tmp_path, monkeypatch):
    wrong = tmp_path / "wrong.pth"
    wrong.write_bytes(b"an unrelated checkpoint")
    accepted = tmp_path / "hpi.pth"
    accepted.write_bytes(b"accepted conversion")
    identity = {"converted_file_sha256": file_sha(accepted), "source_sha256": "official"}
    evidence = {}

    def forbid_load(*args, **kwargs):
        pytest.fail("Candidate selection does not deserialize unverified artifacts")

    monkeypatch.setattr(torch, "load", forbid_load)
    assert resolve_d2_weights([wrong, accepted], [], tmp_path / "new.pth", identity, evidence) == accepted
    assert evidence["method"] == "hpi_rein_equivalent_p16"
    assert [item["sha256"] for item in evidence["candidates"]] == [file_sha(wrong), file_sha(accepted)]
    with pytest.raises(RuntimeError, match="No accepted"):
        resolve_d2_weights([wrong], [wrong], tmp_path / "new.pth", identity, {})


def test_resolve_reconverts_pinned_original_when_no_conversion_matches(tmp_path, monkeypatch):
    original = tmp_path / "original.pth"
    torch.save({"tensor": torch.tensor(3)}, original)
    destination = tmp_path / "regenerated.pth"
    monkeypatch.setattr("tools.check_scs_dual_backbone.convert_state_dict",
                        lambda state: {"tensor": state["tensor"] + 1})
    evidence = {}
    identity = {"converted_file_sha256": "hpi", "source_sha256": file_sha(original)}
    assert resolve_d2_weights([], [original], destination, identity, evidence) == destination
    assert torch.load(destination, weights_only=True)["tensor"].item() == 4
    assert evidence["method"] == "regenerated_rein_p16"
    assert evidence["sha256"] == file_sha(destination)
