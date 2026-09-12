import pytest

from minillm.config import ModelConfig


def test_v01_defaults_match_baseline() -> None:
    config = ModelConfig()

    assert config.vocab_size == 8192
    assert config.d_model == 512
    assert config.n_layers == 8
    assert config.n_heads == 8
    assert config.n_kv_heads == 2
    assert config.context_length == 1024
    assert config.head_dim == 64
    assert config.queries_per_kv == 4


def test_d_model_must_be_divisible_by_n_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=510, n_heads=8)


def test_n_heads_must_be_divisible_by_n_kv_heads() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(n_heads=8, n_kv_heads=3)
