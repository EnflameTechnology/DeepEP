# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

# if we install caps to other dir,
# instead of /opt/tops, we need to set CMAKE_TOPS_COMPILER_TOOLKIT_ROOT
# set(CMAKE_TOPS_COMPILER_TOOLKIT_ROOT   patch_to_caps)
if(DEFINED ENV{TOPS_PATH})
message(STATUS "ENV{TOPS_PATH}: $ENV{TOPS_PATH}")
list(APPEND CMAKE_MODULE_PATH $ENV{TOPS_PATH}/cmake_modules/)
elseif(CMAKE_TOPS_COMPILER_TOOLKIT_ROOT)
message(STATUS "CMAKE_TOPS_COMPILER_TOOLKIT_ROOT: ${CMAKE_TOPS_COMPILER_TOOLKIT_ROOT}")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_TOPS_COMPILER_TOOLKIT_ROOT}/cmake_modules/)
else()
# detect tops in system path
set(DEFAULT_TOPS_PATH /opt/tops)
if(EXISTS ${DEFAULT_TOPS_PATH}/cmake_modules/FindTOPS.cmake)
    message(STATUS "FindTOPS.cmake found in /opt/tops/cmake_modules/FindTOPS.cmake")
    set(CMAKE_TOPS_COMPILER_TOOLKIT_ROOT ${DEFAULT_TOPS_PATH})
    message(STATUS "CMAKE_TOPS_COMPILER_TOOLKIT_ROOT: ${CMAKE_TOPS_COMPILER_TOOLKIT_ROOT}")
    list(APPEND CMAKE_MODULE_PATH ${DEFAULT_TOPS_PATH}/cmake_modules/)
else()
    message(FATAL_ERROR "FindTOPS.cmake not found in /opt/tops/cmake_modules/FindTOPS.cmake")
endif()
endif()
# FIXME: static lib of topsruntime need gelf
set(BUILD_SHARED_LIBS ON)
include(FindTOPS)
enable_language(TOPS)
set(CMAKE_TOPS_STANDARD 17)
update_tops_flags()
set(USE_SYSTEM_NVTX on)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
# find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

# add_library(nvshmem ALIAS nvshmem::nvshmem)
# add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
# add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM ${CMAKE_TOPS_COMPILER_TOOLKIT_ROOT}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR})
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/../2nd/primo_plus_plus/include)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CMAKE_TOPS_COMPILER_TOOLKIT_ROOT}/lib ${NVSHMEM_LIB_DIR})

add_subdirectory(kernels)
# Link CPP and CUDA together
pybind11_add_module(deep_ep_cpp deep_ep.cpp)
target_link_libraries(deep_ep_cpp PRIVATE ${EP_TOPS_LIBRARIES} ${TORCH_LIBRARIES} torch_python torch_gcu)
