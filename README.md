# DeepEP

DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GCU kernels, which are also known as MoE dispatch and combine. The library also supports low-precision operations, including FP8.

To align with the group-limited gating algorithm proposed in the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper, DeepEP offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from ESL domain to RDMA domain. These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks. Additionally, they support compute-unit (SIP) number control.

For latency-sensitive inference decoding, DeepEP includes a set of low-latency kernels with pure RDMA (or ESL) to minimize delays. The library also introduces a hook-based communication-computation overlapping method that does not occupy any compute-unit (SIP) resource.

Notice: the implementation in this library may have some slight differences from the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper.

## Architecture

This GCU port replaces the NVSHMEM + IB-verbs data path of the upstream DeepEP with an Enflame-native stack:

- **Intranode**: data is forwarded through the ESL (Enflame Scale-out Link) fabric, driven by the on-device LARE engine. Each GCU exposes 16 ESL ports; RDMA QPs are set up per port (up to 248 QPs per port) and exchanged between peers at rank initialization (`csrc/ep_init/transport/transport_lare.h`).
- **Internode**: data is forwarded through MORI (GCU-side RDMA), initialized via the IBGDA interface (`ibgdaSetup`, see `csrc/kernels/ibgda.tops`). The host side only performs device enumeration, GDR device selection and topology detection; all QP creation, send/recv and CQE polling happen on the GCU device side.
- **Unified metadata buffer**: QP/MR metadata (`esl_endpoint`, `rdma_ep`, `qp_shared_state`) is published by the host into a unified device buffer (`ibgda_info`), which the kernels consume directly through `EslGdaEngine` / `intra_lare_engine`, avoiding per-call host involvement.

The `csrc/ep_init` layer (bootstrap / graph / channel / transport) is inherited from the Enflame EP communication stack and is responsible for topology detection, channel planning and peer connection only.

## Quick start

## Preceding container & Requirements

Pull and run docker image: `registry-egc.enflame-tech.com/artifacts/deepep:torch2.11.0-TR3.8.106-ubuntu2204`

All requirements are installed in the docker image.

### Development

Build the wheel package locally:

```bash
bash .pipeline/build.sh
```

The generated wheel will be placed in the `dist` directory.

### Installation

Install the wheel after building:

```bash
python3 -m pip install dist/deep_ep*.whl
```

## Network configurations

Internode transfers go through MORI (GCU-side RDMA) on the device. DeepEP (GCU) is fully tested with InfiniBand networks and is theoretically compatible with RoCE as well.

- `IB_GID_INDEX`: GID index used for RoCE, consumed by the IBGDA/MORI QP setup (`csrc/kernels/ibgda.tops`)
- `IB_MERGE_NICS` / `IB_MERGE_VFS` / `IB_PCI_RELAXED_ORDERING` / `IB_ADAPTIVE_ROUTING`: host-side IB device enumeration and capability detection only (`csrc/ep_init/transport/net_ib.cc`); they do not affect the MORI data path

## Interfaces and examples

### Example use in model training or inference prefilling

The normal kernels can be used in model training or the inference prefilling phase (without the backward part) as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of compute-unit (SIP)s to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a TOPS event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GCU's signal to arrive, so this is not compatible with TOPS graph
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
```

Moreover, inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved.

### Example use in inference decoding

The low latency kernels can be used in the inference decoding phase as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import Tuple, Optional

from deep_ep import Buffer

# Communication buffer (will allocate at runtime)
# NOTES: there is no compute-unit (SIP) control API for the low-latency kernels
_buffer: Optional[Buffer] = None


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> Buffer:
    # NOTES: the low-latency mode will consume much more space than the normal mode
    # So we recommend that `num_max_dispatch_tokens_per_rank` (the actual batch size in the decoding engine) should be less than 256
    global _buffer
    num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, group.size(), num_experts)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or not _buffer.low_latency_mode or _buffer.num_rdma_bytes < num_rdma_bytes:
        assert num_experts % group.size() == 0
        _buffer = Buffer(group, 0, num_rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // group.size())
    return _buffer


def low_latency_dispatch(hidden_states: torch.Tensor, topk_idx: torch.Tensor, num_max_dispatch_tokens_per_rank: int, num_experts: int):
    global _buffer

    # Do MoE dispatch, compatible with CUDA graph (but you may restore some buffer status once you replay)
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
                                     async_finish=False, return_recv_hook=True)

    # NOTES: the actual tensor will not be received only if you call `hook()`,
    # it is useful for double-batch overlapping, but **without any SM occupation**
    # If you don't want to overlap, please set `return_recv_hook=False`
    # Later, you can use our GEMM library to do the computation with this specific format
    return recv_hidden_states, recv_expert_count, handle, event, hook


def low_latency_combine(hidden_states: torch.Tensor,
                        topk_idx: torch.Tensor, topk_weights: torch.Tensor, handle: Tuple):
    global _buffer

    # Do MoE combine, compatible with CUDA graph (but you may restore some buffer status once you replay)
    combined_hidden_states, event_overlap, hook = \
        _buffer.low_latency_combine(hidden_states, topk_idx, topk_weights, handle,
                                    async_finish=False, return_recv_hook=True)

    # NOTES: the same behavior as described in the dispatch kernel
    return combined_hidden_states, event_overlap, hook
```

## Roadmap

- [x] Redundant-SIP dispatch (`Buffer.set_num_sms(3)`)
- [x] FP8 dispatch with per-group and per-tensor (external) scales
- [x] UE8M0 scale format for low-latency kernels
- [x] Log-formatted (10-bit) low-latency combine
- [ ] Enable the slave-mode intranode low-latency path (load/store-triggered RDMA write)
- [ ] `enable_shrink` for vLLM 0.26 compatibility (API reserved, not implemented yet)

## Notices

- Some constructor arguments are accepted for API compatibility with the upstream version but are not effective on GCU: `num_qps_per_rank`, `allow_nvlink_for_low_latency_mode`, `allow_mnnvl`. The QP number is auto-adapted from the channel count (`kMasterChannelCount * 2`).
- `low_latency_dispatch` returns `recv_count` and `handle` as int32 tensors (the upstream CUDA version returns int64).
- When switching from the normal mode to the low-latency mode on the same `Buffer`, call `clean_low_latency_buffer` first.
- FP8 dispatch supports two scale modes: `round_scale` (power-of-two scales) and `use_ue8m0` (UE8M0 scale format). A single per-tensor scale can be supplied with `fp8_quant_scale` (mutually exclusive with `use_ue8m0`).
- The low-latency kernels support a hook-based receiving interface (`return_recv_hook=True`) so that the RDMA traffic happens in the background without occupying GCU SMs; useful for double-batch overlapping.

## License

This code repository is released under [the MIT License](LICENSE). Third-party dependencies (e.g., MORI, TopsPlatform) are subject to their own licenses.

## Citation

If you use this codebase or otherwise find our work valuable, please cite:

```bibtex
@misc{deepep2026,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Yizhou Li and Zhaoyuan Ye and Huilung Wang},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/EnflameTechnology/DeepEP}},
}
```