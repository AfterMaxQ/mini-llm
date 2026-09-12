from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    context_length: int = 1024

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def queries_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads
