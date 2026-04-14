#!/bin/sh
# Install shot-boundary CLI from GitHub Releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/AIGC-Hackers/shot-boundary-zig/main/scripts/install.sh | sh
#   curl -fsSL <url> | sh -s -- --version v0.1.3
#
# Environment:
#   SHOT_BOUNDARY_HOME  Install directory (default: ~/.shot-boundary)

set -eu

REPO="AIGC-Hackers/shot-boundary-zig"
INSTALL_DIR="${SHOT_BOUNDARY_HOME:-${HOME}/.shot-boundary}"
BIN_LINK_DIR="${HOME}/.local/bin"

main() {
    parse_args "$@"
    detect_platform
    resolve_version
    download
    verify
    install
    link
    print_success
}

parse_args() {
    VERSION=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --version|-v)
                [ $# -ge 2 ] || die "--version requires a value"
                VERSION="$2"; shift 2 ;;
            --help|-h) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
}

detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "${OS}" in
        Linux) ;;
        *) die "unsupported OS: ${OS} (only Linux is currently supported)" ;;
    esac

    case "${ARCH}" in
        x86_64|amd64)   PLATFORM="linux-x64" ;;
        aarch64|arm64)   PLATFORM="linux-aarch64" ;;
        *) die "unsupported architecture: ${ARCH}" ;;
    esac
}

resolve_version() {
    if [ -z "${VERSION}" ]; then
        VERSION="$(curl -fsSL -o /dev/null -w '%{url_effective}' \
            "https://github.com/${REPO}/releases/latest")"
        VERSION="${VERSION##*/}"
    fi
    case "${VERSION}" in
        v*) ;;
        *)  VERSION="v${VERSION}" ;;
    esac
    VERSION_NUM="${VERSION#v}"
}

download() {
    TARBALL="shot-boundary-${VERSION_NUM}-${PLATFORM}.tar.gz"
    BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "${TMPDIR}"' EXIT

    echo "Downloading shot-boundary ${VERSION} (${PLATFORM})..."
    curl -fSL "${BASE_URL}/${TARBALL}"      -o "${TMPDIR}/${TARBALL}"
    curl -fsSL "${BASE_URL}/${TARBALL}.sha256" -o "${TMPDIR}/${TARBALL}.sha256"
}

verify() {
    echo "Verifying checksum..."
    EXPECTED="$(awk '{print $1}' "${TMPDIR}/${TARBALL}.sha256")"
    ACTUAL="$(sha256sum "${TMPDIR}/${TARBALL}" | awk '{print $1}')"
    [ "${EXPECTED}" = "${ACTUAL}" ] || die "checksum mismatch: expected ${EXPECTED}, got ${ACTUAL}"
}

install() {
    echo "Installing to ${INSTALL_DIR}..."
    rm -rf "${INSTALL_DIR}"
    mkdir -p "${INSTALL_DIR}"
    tar -xzf "${TMPDIR}/${TARBALL}" -C "${INSTALL_DIR}" --strip-components=1
}

link() {
    mkdir -p "${BIN_LINK_DIR}"
    ln -sf "${INSTALL_DIR}/bin/shot-boundary" "${BIN_LINK_DIR}/shot-boundary"
}

print_success() {
    echo ""
    echo "shot-boundary ${VERSION} installed to ${INSTALL_DIR}"
    echo "Symlink: ${BIN_LINK_DIR}/shot-boundary -> ${INSTALL_DIR}/bin/shot-boundary"

    case ":${PATH}:" in
        *":${BIN_LINK_DIR}:"*)
            ;;
        *)
            echo ""
            echo "NOTE: ${BIN_LINK_DIR} is not in your PATH. Add it:"
            echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
            ;;
    esac

    echo ""
    echo "Verify: shot-boundary env"
}

usage() {
    cat <<'EOF'
Install shot-boundary CLI from GitHub Releases.

Usage:
  curl -fsSL <url> | sh
  curl -fsSL <url> | sh -s -- [options]

Options:
  --version, -v <version>  Install a specific version (default: latest)
  --help, -h               Show this help

Environment:
  SHOT_BOUNDARY_HOME       Install directory (default: ~/.shot-boundary)
EOF
}

die() {
    echo "error: $1" >&2
    exit 1
}

main "$@"
