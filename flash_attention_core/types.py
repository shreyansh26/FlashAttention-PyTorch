from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ForwardResult:
    out: torch.Tensor
    lse: torch.Tensor
    normalizers: torch.Tensor | None = None
    row_max: torch.Tensor | None = None
    saved_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackwardResult:
    dQ: torch.Tensor
    dK: torch.Tensor
    dV: torch.Tensor
    debug_state: dict[str, Any] = field(default_factory=dict)
