# -*- coding: utf-8 -*-
import os
import subprocess
import setuptools
import importlib
import importlib.util
import shutil
from pathlib import Path
from setuptools import find_packages

from tops_extension import TopsBuildExtension
from tops_extension.torch import TopsTorchExtension
from distutils.dist import Distribution as DistutilsDistribution
import torch
import torch_gcu

base_dir = os.environ.get('BASE_DIR', '')
extra_compile_option = os.environ.get('COMPILE_CXX_FLAGS', '')
extra_link_option = os.environ.get('LINK_FLAGS', '')
extra_link_libs = os.environ.get('LINK_LIBS', '')
extra_build_type = os.environ.get('CMAKE_BUILD_TYPE', '')
extra_topscc_options = os.environ.get('TOPSCC_EXTRA_FLAGS', '')

extra_compile_flags = extra_compile_option.split('\\ ')
extra_compile_flags = [f for f in extra_compile_flags if f]
extra_link_flags = extra_link_option.split()
extra_link_flags = [l for l in extra_link_flags if l]
extra_link_libs = extra_link_libs.split()
extra_link_libs = [l for l in extra_link_libs if l]

_abs_file_ = os.path.abspath(__file__)
source_root_dir = os.path.dirname(_abs_file_)

# check ccache | sccache tool, but torch.utils cannot support for now.
def get_cache_tool() -> str:
    """ Check if which cache tool(ccache/sccache) is available

    Use sccache is 'ENABLE_SCCACHE' is set.

    """
    cache_tool = ''
    if os.environ.get('ENABLE_SCCACHE', '0').lower()\
                            in ('true', '1', 'on', 't'):
        cache_tool = shutil.which("sccache")
    if not cache_tool:
        cache_tool = shutil.which("ccache")
    return cache_tool if cache_tool else ''

def get_pkg_install_dir(pkg_name: str) -> str:
    print("get_pkg_install_dir: {}".format(pkg_name))
    try:
        spec = importlib.util.find_spec(pkg_name)
        print("spec: {}".format(spec))
        if spec and spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin))
        else:
            # raise("{} is not installed".format(pkg_name))
            print("Error: {} is not installed".format(pkg_name))
            return None
    except Exception as e:
        # raise("{} is not installed".format(pkg_name))
        print("Error: {} is not installed".format(pkg_name))
        return None

torch_install_dir = get_pkg_install_dir("torch")
torch_gcu_install_dir = get_pkg_install_dir("torch_gcu")

aten_include_dir = os.path.join(torch_install_dir, "include/ATen")
torch_include_dir = os.path.join(torch_install_dir, "include")
torch_api_include_dir = os.path.join(torch_install_dir, "include/torch/csrc/api/include")
torch_gcu_include_dir = os.path.join(torch_gcu_install_dir, "include")

top_runtime_include_dir = [ base_dir + "/opt/tops/include" ]
top_aten_include_dir = [ base_dir + "/usr/include/gcu" ]
include_dirs = [ base_dir+"/usr/include" ]
if base_dir:
    top_runtime_include_dir.append("/opt/tops/include")
    top_aten_include_dir.append("/usr/include/gcu")
    include_dirs.append("/usr/include")

def get_build_ext_cmd():
    # Create a distribution that parses command line arguments
    dist = DistutilsDistribution()
    dist.parse_command_line()

    # Create build commands with the parsed options
    # Use TopsBuildExtension to get the actual paths that will be used
    build_ext_cmd = TopsBuildExtension(dist)
    build_ext_cmd.initialize_options()
    build_ext_cmd.finalize_options()

    return build_ext_cmd


DEEP_EP_SRC_DIR = os.getenv('DEEP_EP_SRC_DIR', source_root_dir)

sys_include_dirs = [aten_include_dir, torch_include_dir, torch_api_include_dir, torch_gcu_include_dir]
sys_include_dirs.extend(top_runtime_include_dir)
sys_include_dirs.extend(top_aten_include_dir)

torch_lib_dir = os.path.join(torch_install_dir, "lib")
torch_gcu_lib_dir = os.path.join(torch_gcu_install_dir, "lib")
common_lib_dir = [
    base_dir + '/usr/lib',
    base_dir + '/opt/tops/lib']
if base_dir:
    common_lib_dir.extend(["/usr/lib", "/opt/tops/lib"])
library_dirs = []
library_dirs.extend([torch_lib_dir, torch_gcu_lib_dir])
library_dirs.extend(common_lib_dir)
libraries = extra_link_libs
# libraries.extend(["c10", "torch", "torch_cpu", "torch_python", "torch_gcu", "topsaten"])
libraries.extend(["torch", "torch_gcu"])
libraries.extend(["mori-gcu"]) # support mori for ibgda
libraries.extend(["topstx"])   # topstx CPU tracing (deep_ep profiler domain)
cxx_flags = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-Werror"
]

cxx_flags.extend(sum([['-isystem', str(path)] for path in sys_include_dirs], []))
cxx_flags.extend(["-DENABLE_MORI_GCU"]) # support mori for ibgda
cxx_flags.extend(["-DENABLE_LARE_GDA_TESTS"]) # enable LARE GDA test functions

topscc_flags = cxx_flags.copy()
topscc_flags.extend([
    "-Wno-unused-parameter"
])
cxx_flags.extend(extra_compile_flags)
if extra_topscc_options:
    extra_topscc_flags = extra_topscc_options.split()
    topscc_flags.extend(extra_topscc_flags)

def get_features_args():
    features_args = []
    return features_args

# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(DEEP_EP_SRC_DIR):
    path = Path(DEEP_EP_SRC_DIR).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')

if __name__ == '__main__':

    TORCH_GCU_PATH = os.getenv('TORCH_TORCH_GCU_PATHPATH', '')
    TORCH_VERSION = os.getenv('TORCH_VERSION', '')
    if TORCH_VERSION == '':
        import torch
        TORCH_VERSION = torch.__version__.split("+")[0]
    if TORCH_GCU_PATH == '':
        import torch_gcu
        TORCH_GCU_PATH = torch_gcu.__path__[0]

    # cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
    #              '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes',
    #              '-isystem', '/opt/tops/include']
    # library_dirs = ['/opt/tops/lib']
    # if TORCH_GCU_PATH != '':
    #     cxx_flags.extend(['-isystem', TORCH_GCU_PATH+'/include'])
    #     library_dirs.append(TORCH_GCU_PATH+'/lib')
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = []
    sources_org = ['csrc/deep_ep.cpp',
                'csrc/ep_init/net.cc',
                'csrc/ep_init/mnlare.cc',
                'csrc/ep_init/ep_init.cc',
                'csrc/ep_init/ep_bootstrap.cc',
                'csrc/ep_init/channel.cc',
                'csrc/ep_init/transport.cc',
                'csrc/ep_init/graph/connect.cc',
                'csrc/ep_init/graph/graph.cc',
                'csrc/ep_init/graph/detector.cc',
                'csrc/ep_init/graph/paths.cc',
                'csrc/ep_init/graph/search.cc',
                'csrc/ep_init/graph/topo.cc',
                'csrc/ep_init/graph/xml.cc',

                'csrc/ep_init/misc/argcheck.cc',
                'csrc/ep_init/misc/debug.cc',
                'csrc/ep_init/misc/gcu_info.cc',
                'csrc/ep_init/misc/param.cc',
                'csrc/ep_init/misc/utils.cc',
                'csrc/ep_init/misc/ibvsymbols.cc',
                'csrc/ep_init/misc/ibvwrap.cc',
                'csrc/ep_init/misc/mlxwrap.cc',
                'csrc/ep_init/misc/efmlwrap.cc',
                'csrc/ep_init/misc/socket.cc',

                'csrc/ep_init/transport/p2p.cc',
                'csrc/ep_init/transport/net_ib.cc',
                'csrc/ep_init/transport/lare_roce_ctxt.cc',

                'csrc/kernels/runtime.tops',
                'csrc/kernels/layout.tops',
                'csrc/kernels/ibgda.tops',
                'csrc/kernels/intranode.tops',
                'csrc/kernels/internode.tops',
                'csrc/kernels/internode_ll.tops',
                'csrc/kernels/intranode_ll.tops',
                'csrc/kernels/intranode_ll_slave.tops',
                'csrc/kernels/delay.tops']

    current_dir = os.getcwd()
    if current_dir != DEEP_EP_SRC_DIR:
        for source in sources_org:
            sources.append(os.path.join(DEEP_EP_SRC_DIR, source))
    else:
        sources = sources_org
    include_dirs.append(os.path.join(DEEP_EP_SRC_DIR, 'csrc/ep_init/include'))
    include_dirs.append(os.path.join(DEEP_EP_SRC_DIR, 'csrc/kernels'))
    include_dirs.append(os.path.join(DEEP_EP_SRC_DIR, 'csrc/'))

    # Put them together
    extra_compile_args = {
            "cxx": cxx_flags + get_features_args(),
            "topscc": [
                "-O3",
                "-std=c++17",
                "-arch", "gcu400",
                "-arch", "gcu410",
            ] + topscc_flags + get_features_args(),

    }
    # Summary
    print('Build summary:')
    print(' > Sources: {}'.format(sources))
    print(' > Includes: {}'.format(include_dirs))
    print(' > Libraries: {}'.format(libraries))
    print(' > Compilation flags: {}'.format(extra_compile_args))
    print(' > Link flags: {}'.format(extra_link_flags))

    # noinspection PyBroadException
    v_revision = os.getenv('PACKAGE_VERSION', '')
    if not v_revision:
        try:
            with open(os.path.join(source_root_dir, '.version'), 'r') as f:
                v_revision = f.read().strip()
        except: v_revision = ''
        if v_revision:
            try:
                cmd = ['git', 'rev-parse', '--short', 'HEAD']
                g_date = subprocess.check_output(cmd).decode('ascii').rstrip()
                v_revision += '.' + g_date
            except: pass

    v_torch_version = "torch." + TORCH_VERSION + '.gcu' if TORCH_VERSION else ''
    GCU_SANITIZER = os.getenv('GCU_SANITIZER', '')
    if GCU_SANITIZER and GCU_SANITIZER.lower() != 'off' and GCU_SANITIZER.lower() != 'none':
        v_torch_version += '.{}'.format(GCU_SANITIZER)
    SANITIZER = os.getenv('SANITIZER', '')
    if SANITIZER and SANITIZER.lower() != 'off' and SANITIZER.lower() != 'none':
        v_torch_version += '.{}'.format(SANITIZER)
    v_version = v_torch_version + '.' + v_revision if v_revision else v_torch_version
    version = '1.2.1'
    if v_version: version += '+' + v_version

    relative_path = os.path.relpath(source_root_dir, os.getcwd())
    packages = find_packages(
        where=source_root_dir,
        include=['deep_ep']
    )
    package_dir = {}

    for p in packages:
        package_dir[p] = os.path.join(relative_path, p.replace('.', '/'))

    setuptools.setup(
        name='deep_ep',
        version=version,
        ext_modules=[
            TopsTorchExtension(
                name='deep_ep_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_flags,
                libraries=libraries,
            )
        ],
        cmdclass={
            'build_ext': TopsBuildExtension
        },
        package_dir=package_dir,
        packages=packages,
    )
