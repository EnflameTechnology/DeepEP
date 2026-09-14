#pragma once

#include "configs.h"
#include "exception.h"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_blocks, num_threads, num_sub_threads, stream) \
    topsLaunchConfig_t cfg = {nullptr, (num_threads), 0, (num_blocks), 0, 0, \
                              stream}; \
    topsLaunchAttribute attr[2]; \
    attr[0].id = topsLaunchAttributeThreadDimension; \
    attr[0].val.ThreadDim.x = (num_sub_threads); \
    attr[0].val.ThreadDim.y = 1; \
    attr[0].val.ThreadDim.z = 1; \
    attr[1].id = topsLaunchAttributeCooperative; \
    attr[1].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 2;
#endif

// Launch on redundant SIP (SIP Lite). Same grid/thread layout as
// SETUP_LAUNCH_CONFIG,
// but prefer topsSchedulingPolicyRedundant.
#ifndef SETUP_LAUNCH_CONFIG_REDUNDANT
#define SETUP_LAUNCH_CONFIG_REDUNDANT(num_blocks, num_threads, \
                                      num_sub_threads, stream) \
    topsLaunchConfig_t cfg = {nullptr, (num_threads), 0, (num_blocks), 0, 0, \
                              stream}; \
    topsLaunchAttribute attr[3]; \
    attr[0].id = topsLaunchAttributeThreadDimension; \
    attr[0].val.ThreadDim.x = (num_sub_threads); \
    attr[0].val.ThreadDim.y = 1; \
    attr[0].val.ThreadDim.z = 1; \
    attr[1].id = topsLaunchAttributeCooperative; \
    attr[1].val.cooperative = 1; \
    attr[2].id = topsLaunchAttributeSchedulingPolicy; \
    attr[2].val.SchedulingPolicyPreference = topsSchedulingPolicyRedundant; \
    cfg.attrs = attr; \
    cfg.numAttrs = 3;
#endif

// remove Programmatic Stream Serialization for now
// attr[2].id = topsLaunchAttributeProgrammaticStreamSerialization; 
// attr[2].val.programmaticStreamSerializationAllowed = 1; 

// attr[1].id = topsLaunchAttributeSchedulingPolicy;
// attr[1].val.SchedulingPolicyPreference = topsSchedulingPolicyRedundant;

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) TOPS_CHECK(topsLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel) \
EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 32: case_macro(32); \
        case 64: case_macro(64); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_rdma_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 6: case_macro(6); \
        case 8: case_macro(8); \
        case 12: case_macro(12); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case TOPS_R_16BF: case_macro(tops::__ef_bfloat16); \
        default: EP_HOST_ASSERT(false and "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2048: case_macro(2048); \
        case 2560: case_macro(2560); \
        case 4096: case_macro(4096); \
        case 5120: case_macro(5120); \
        case 6144: case_macro(6144); /* For qwen3 coder */ \
        case 7168: case_macro(7168); \
        case 8192: case_macro(8192); \
        default: EP_HOST_ASSERT(false and "Unsupported hidden"); \
    } while (false)

/* BF16 values: 2048..8192; FP8 values: hidden/2 = 1024..4096 */
#define SWITCH_INTRA_HIDDEN(case_macro) \
    switch (hidden) { \
        case 1024: case_macro(1024); \
        case 1280: case_macro(1280); \
        case 2048: case_macro(2048); \
        case 2560: case_macro(2560); \
        case 3072: case_macro(3072); \
        case 3584: case_macro(3584); \
        case 4096: case_macro(4096); \
        case 5120: case_macro(5120); \
        case 6144: case_macro(6144); \
        case 7168: case_macro(7168); \
        case 8192: case_macro(8192); \
        default: EP_HOST_ASSERT(false and "Unsupported intra hidden"); \
    } while (false)

    // ranks 32/40/48: may using 1 SIP with {1, x, 8} for best performance sync inside of combine
#define CALCULATE_LAUNCH_CONFIG_BY_RANKS(NRANKS) \
    ([&]() -> std::tuple<int, int, int> { \
        switch (NRANKS) { \
            case 2: return {1, 6, NRANKS}; \
            case 4: return {1, 6, NRANKS}; \
            case 8: return {2, 6, 8}; \
            case 16: return {2, 6, 8}; \
            case 32: return {2, 6, 8}; \
            case 40: return {2, 4, 5}; \
            case 48: return {2, 3, 8}; \
            case 56: return {2, 4, 7}; \
            case 64: return {4, 6, 8}; \
            default: EP_HOST_ASSERT(false && "Unsupported num_ranks"); \
        } \
    }())

#define CALCULATE_LAUNCH_CONFIG_BY_SMS(NSMS, NRANKS) \
    ([&]() -> std::tuple<int, int, int> { \
        switch (NSMS) { \
            case 12: return {1, 6, NRANKS}; \
            case 20: return {2, 5, NRANKS}; \
            case 24: return {2, 6, NRANKS}; \
            default: EP_HOST_ASSERT(false && "Unsupported NSMS"); \
        } \
    }())

// Combine uses 2 SIPs per channel (sender + receiver), so doubles the SIP count.
// NSMS is the logical channel count; actual SIPs launched = 2 * channels.
#define CALCULATE_COMBINE_LAUNCH_CONFIG_BY_SMS(NSMS, NRANKS) \
    ([&]() -> std::tuple<int, int, int> { \
        switch (NSMS) { \
            case 12: return {2, 6, NRANKS}; \
            case 20: return {4, 5, NRANKS}; \
            case 24: return {4, 6, NRANKS}; \
            default: EP_HOST_ASSERT(false && "Unsupported NSMS for combine"); \
        } \
    }())
