# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""FLA BF16 diagnostic with hard assertions, separate from native acceptance.

Run explicitly: pytest -s tests/python/gdn_fla_bf16_diagnostic.py
FLA 0.5.2 BF16 weak/zero-decay failures remain failures, not xfail or warnings.
"""
import pytest
import torch

from torch_gdn_test import ISSUE29_SHAPES, gfx936, _compare_native_and_fla


@pytest.mark.skipif(not gfx936(), reason="native GDN requires gfx936")
@pytest.mark.parametrize("K,V", ISSUE29_SHAPES + [(128, 128)])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("gate", ["normal", "weak", "zero", "boundary"])
def test_native_and_fla_against_fp64(monkeypatch, K, V, norm, gate):
    _compare_native_and_fla(monkeypatch, K, V, norm, gate, torch.bfloat16)
