# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""flash-train PyTorch bindings."""

import ctypes
import os

# The extension NEEDs libtorch.so, which only enters the process with torch.
import torch

# Wheels built with FTRAIN_BUILD_SHARED_LIBS=ON ship libflash_train.so in
# the package root; loading it by absolute path pins the bundled copy over
# anything on LD_LIBRARY_PATH. Static wheels ship no file and skip this.
_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_native_library = os.path.join(_package_root, "libflash_train.so")
if os.path.exists(_native_library):
    ctypes.CDLL(_native_library)

try:
    from ._flash_train_torch import gemm
    from .gdn2 import chunk_gdn2
    from .gdn2_reference import gdn2_reference
    from .kda import chunk_kda, kda_reference, kimi_delta_attention
    from .gdn import chunk_gated_delta_rule, gdn_reference
except ModuleNotFoundError as exc:
    # This shell ships even in wheels built without the binding; a missing
    # extension module is a build problem, not a missing dependency.
    if exc.name != "flash_train.torch._flash_train_torch":
        raise
    raise RuntimeError(
        "this wheel was built without the torch binding; rebuild it with "
        "FTRAIN_BUILD_TORCH_BINDINGS=ON"
    ) from exc

__all__ = ["gemm", "chunk_kda", "kimi_delta_attention", "kda_reference", "chunk_gated_delta_rule", "gdn_reference", "chunk_gdn2", "gdn2_reference"]
