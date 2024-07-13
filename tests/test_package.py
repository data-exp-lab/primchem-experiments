from __future__ import annotations

import importlib.metadata

import primchem_experiments as m


def test_version():
    assert importlib.metadata.version("primchem_experiments") == m.__version__
