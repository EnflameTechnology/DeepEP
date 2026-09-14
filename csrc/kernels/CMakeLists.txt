function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD_REQUIRED ON
            CXX_STANDARD 17
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_deep_ep_library(runtime_cuda runtime.cu)
add_deep_ep_library(layout_cuda layout.cu)
add_deep_ep_library(intranode_cuda intranode.cu)
add_deep_ep_library(internode_cuda internode.cu)
add_deep_ep_library(internode_ll_cuda internode_ll.cu)

# Later, we should link all libraries in `EP_TOPS_LIBRARIES`
set(EP_TOPS_LIBRARIES runtime_cuda layout_cuda intranode_cuda internode_cuda internode_ll_cuda PARENT_SCOPE)
