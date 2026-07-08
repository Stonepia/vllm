# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion Kernel test utils"""

import pytest

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.platforms import current_platform


def skip_if_platform_unsupported(op_name: str):
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        # is_cuda_alike() covers CUDA + ROCm (mirrors torch.cuda.is_available());
        # is_xpu() adds Intel XPU. Extend here as more platforms gain configs.
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            pytest.skip(f"No supported accelerator available for {op_name} kernel")

        platform = get_canonical_gpu_name()

        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs(op_name, platform)
        if len(configs) == 0:
            pytest.skip(f"Current GPU platform not supported for {op_name} kernel")

    except (ImportError, RuntimeError, KeyError):
        pytest.skip(f"Error detecting platform support for {op_name} kernel")
