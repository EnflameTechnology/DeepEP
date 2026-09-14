#!/bin/bash
# Copyright 2024 Enflame. All Rights Reserved.
#
set -eu -o pipefail
# set -x
SCRIPT_DIR=$(dirname $(realpath $0))
project_dir=${SCRIPT_DIR}/..
BUILD_ROOT_DIR=$(pwd)
ARCH=$(uname -m)

# run in docker container
# registry-egc.enflame-tech.com/artifacts/deepep:torch2.11.0-TR3.8.106-ubuntu2204

rm -rf ${project_dir}/dist ${project_dir}/build ${project_dir}/deep_ep.egg-info

function ci_build() {
  # cd ${project_name}

  sed -i 's/import tops_extension.torch._inductor/#import tops_extension.torch._inductor/g; /import tops_extension.torch._inductor/a \    pass' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
  cd ${project_dir}
  python3 setup.py bdist_wheel
  cd -
  sed -i 's/#import tops_extension.torch._inductor/import tops_extension.torch._inductor/g' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
  sed -i '5,12{/pass/d}' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
}

function install() {
  # cd ${project_name}

  sed -i 's/import tops_extension.torch._inductor/#import tops_extension.torch._inductor/g; /import tops_extension.torch._inductor/a \    pass' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
  cd ${project_dir}
  python3 setup.py install
  cd -
  sed -i 's/#import tops_extension.torch._inductor/import tops_extension.torch._inductor/g' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
  sed -i '5,12{/pass/d}' /usr/local/lib/python3.10/dist-packages/tops_extension/torch/__init__.py
}

function main() {
  $build_job_name
}

# export project_name=${project_name:-"attention"}

build_job_name=${1:-ci_build}

main "$@"
exit $?
         