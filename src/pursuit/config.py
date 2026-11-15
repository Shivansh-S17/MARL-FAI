"""Configuration loading (YAML, deep-merged over configs/base.yaml)."""
from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
MAP_DIR = ROOT / "maps"


class Cfg(dict):
    """dict with attribute access; nested dicts are wrapped on construction."""

    def __init__(self, d=None):
        super().__init__()
        for k, v in (d or {}).items():
            self[k] = Cfg(v) if isinstance(v, dict) else v

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Cfg) else copy.deepcopy(v)) for k, v in self.items()}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides: dict | None = None) -> Cfg:
    with open(CONFIG_DIR / "base.yaml") as f:
        d = yaml.safe_load(f)
    if path:
        with open(path) as f:
            d = deep_merge(d, yaml.safe_load(f) or {})
    if overrides:
        d = deep_merge(d, overrides)
    return Cfg(d)


def parse_budget(b) -> Fraction:
    if isinstance(b, Fraction):
        return b
    return Fraction(str(b))
