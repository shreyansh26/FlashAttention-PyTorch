from __future__ import annotations

from . import fa1, fa2, fa3, fa4

VERSION_REGISTRY = {
    "fa1": fa1,
    "fa2": fa2,
    "fa3": fa3,
    "fa4": fa4,
}


def get_version_module(version: str):
    try:
        return VERSION_REGISTRY[version]
    except KeyError as exc:
        raise ValueError(f"Unknown FlashAttention version: {version}") from exc


def list_versions() -> list[str]:
    return list(VERSION_REGISTRY)
