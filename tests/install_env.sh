#!/bin/bash
#
# Install deep_ep Python environment: dependencies and wheel package.
# Requires: python3, sudo (for system-wide pip install).
#
# Usage:
#   bash install_env.sh           # Quiet pip output (default)
#   bash install_env.sh --debug   # Show full pip install logs
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_step()  { echo -e "${YELLOW}[STEP]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

DEBUG=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug) DEBUG=1; shift ;;
        -h|--help)
            echo "Usage: $0 [--debug|-d]"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Usage: $0 [--debug|-d]"
            exit 1
            ;;
    esac
done

run_pip() {
    local pip_args=("$@")
    local index_args=()

    if [ -n "${PIP_INDEX_URL:-}" ]; then
        index_args+=( -i "${PIP_INDEX_URL}" )
    fi
    if [ -n "${PIP_TRUSTED_HOST:-}" ]; then
        index_args+=( --trusted-host "${PIP_TRUSTED_HOST}" )
    fi

    if [ "$DEBUG" -eq 1 ]; then
        sudo python3 -m pip "${pip_args[@]}" "${index_args[@]}"
    else
        local log_file
        log_file="$(mktemp /tmp/deepep_pip_XXXXXX.log)"
        if sudo python3 -m pip "${pip_args[@]}" "${index_args[@]}" > "$log_file" 2>&1; then
            rm -f "$log_file"
        else
            echo -e "${RED}[ERROR]${NC} pip install failed:"
            cat "$log_file"
            log_info "Re-run with --debug to stream pip output live."
            rm -f "$log_file"
            exit 1
        fi
    fi
}

TOTAL_STEPS=2
STEP=0

next_step() {
    STEP=$((STEP + 1))
    echo ""
    log_step "[$STEP/$TOTAL_STEPS] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/.. && pwd)"

echo ""
echo "============================================"
echo "  DeepEP Environment Installer"
echo "============================================"
log_info "Working directory: $REPO_ROOT"
log_info "Python:          $(python3 --version 2>&1 || echo 'not found')"

WHEELS=($REPO_ROOT/dist/deep_ep*.whl)
if [ ! -f "${WHEELS[0]}" ]; then
    echo -e "${RED}[ERROR]${NC} No deep_ep wheel found in $REPO_ROOT/dist/"
    log_info "Build the wheel first, e.g.: bash .pipeline/build.sh build"
    exit 1
fi

REQUIREMENTS="$REPO_ROOT/requirements.txt"
if [ -f "$REQUIREMENTS" ]; then
    next_step "Install Python dependencies"
    run_pip install -r "$REQUIREMENTS"
    log_ok "Dependencies installed"
fi

next_step "Install deep_ep wheel"
log_info "Wheel package(s):"
ls -lh "${WHEELS[@]}"
echo ""
log_warn "You may be prompted for your sudo password again."
run_pip install "${WHEELS[@]}"
log_ok "deep_ep installed: $(python3 -c 'import deep_ep; print(deep_ep.__file__)' 2>/dev/null || echo '(import check skipped)')"

echo ""
echo "============================================"
echo -e "  ${GREEN}Environment installation finished.${NC}"
echo "============================================"
echo ""
          