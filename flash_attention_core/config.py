from dataclasses import dataclass


@dataclass(frozen=True)
class FlashAttentionConfig:
    """Shared knobs for the simplified educational implementations."""

    block_size_q: int = 1024
    block_size_kv: int = 1024
    num_stages: int = 2
    fp8: bool = False
    keep_debug_state: bool = True
