import torch

from .utils import EventOverlap
from .buffer import Buffer

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config

# Export LARE GDA test functions (enabled by ENABLE_LARE_GDA_TESTS)
# noinspection PyUnresolvedReferences
try:
    from deep_ep_cpp import (
        test_simple_send,
        test_multi_qp,
        test_contention
    )
except ImportError:
    # Test functions not available (compiled without ENABLE_LARE_GDA_TESTS)
    pass
