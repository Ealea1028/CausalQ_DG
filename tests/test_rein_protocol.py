import pytest

from tools.inspect_rein_protocol import protocol_fields


def test_protocol_inventory_preserves_effective_optimizer_and_loaders():
    config = {key: {"fixture": key} for key in
              ("train_dataloader", "val_dataloader", "test_dataloader", "optim_wrapper",
               "param_scheduler", "train_cfg", "val_evaluator")}
    config["optim_wrapper"] = {"paramwise_cfg": {"norm_decay_mult": 0},
                              "optimizer": {"lr": 1e-4}}
    config["model"] = {"type": "EncoderDecoder", "test_cfg": {"mode": "slide"},
                       "decode_head": {"num_classes": 19}}
    result = protocol_fields(config)
    assert result["optim_wrapper"] == config["optim_wrapper"]
    assert result["model_contract"]["test_cfg"] == {"mode": "slide"}
    assert result["head_num_classes"] == 19
    assert result["val_dataloader"] == config["val_dataloader"]


def test_protocol_inventory_rejects_partial_config():
    with pytest.raises(ValueError, match="Incomplete"):
        protocol_fields({"model": {}})
