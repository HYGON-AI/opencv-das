#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Hygon Information Technology Co., Ltd.

set -euo pipefail

: "${DTK_PKG_URL:?DTK_PKG_URL is required}"
: "${RESOURCE_SERVER_URL:?RESOURCE_SERVER_URL is required}"

if [[ -e /opt/dtk ]]; then
  echo "/opt/dtk already exists; the CI container must provide a clean DTK install location" >&2
  exit 1
fi

package_manager=""
if command -v dnf >/dev/null 2>&1; then
  package_manager=dnf
elif command -v yum >/dev/null 2>&1; then
  package_manager=yum
else
  echo "A yum-compatible package manager is required" >&2
  exit 1
fi

"${package_manager}" install -y \
  gcc \
  gcc-c++ \
  git \
  make \
  ninja-build \
  pkgconf-pkg-config \
  wget \
  which

dtk_archive="$(mktemp --tmpdir=/opt DTK-package.XXXXXX.tar.gz)"
cleanup() {
  rm -f "${dtk_archive}"
}
trap cleanup EXIT

wget --quiet --tries=3 --timeout=30 --output-document="${dtk_archive}" "${DTK_PKG_URL}"
tar -xzf "${dtk_archive}" -C /opt

extracted_dirs=(/opt/dtk-*)
if [[ "${#extracted_dirs[@]}" -ne 1 || ! -d "${extracted_dirs[0]}" ]]; then
  echo "Expected exactly one DTK directory after extracting ${DTK_PKG_URL}" >&2
  exit 1
fi
mv "${extracted_dirs[0]}" /opt/dtk
test -f /opt/dtk/env.sh

aicc_installer="$(mktemp)"
trap 'rm -f "${dtk_archive}" "${aicc_installer}"' EXIT
wget --quiet --tries=3 --timeout=30 \
  --output-document="${aicc_installer}" \
  "${RESOURCE_SERVER_URL}/ai_cc/Nightly/hcu_llvm_installer.sh"
bash "${aicc_installer}" --major "${AICC_MAJOR_VERSION:-1.0.0}"
