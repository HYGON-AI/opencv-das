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

# Avoid conflicting system FFmpeg development packages; the codec SDK/DTK
# supplies the versions used by this build. These packages may be absent.
"${package_manager}" remove -y libavcodec-devel libavutil-devel libavfilter-devel || true

"${package_manager}" install -y \
  gcc gcc-c++ git git-lfs make ninja-build pkgconf-pkg-config wget which

for optional_package in \
  python3-devel libX11-devel libXrandr-devel gtk3-devel libv4l-devel libXv-devel \
  x264-devel libjpeg-turbo-devel libpng-devel libtiff-devel gcc-gfortran \
  openexr-devel atlas-devel hdf5-devel openblas-devel eigen3-devel \
  ogre-devel libXaw-devel bullet-devel bullet-extras-devel; do
  "${package_manager}" install -y "${optional_package}" || \
    echo "Optional package unavailable: ${optional_package}" >&2
done

python3 -m pip install --disable-pip-version-check \
  wheel==0.45.1 scikit-build numpy==1.26.4

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

maccodec_url="http://42.228.13.241:18000/customized/dali/MacCodecSDK-dtk-8faa7a3c-26a65983.tar.gz"
maccodec_archive="$(mktemp --tmpdir=/opt MacCodecSDK.XXXXXX.tar.gz)"
maccodec_tmp="$(mktemp -d --tmpdir=/opt MacCodecSDK.XXXXXX)"
trap 'rm -f "${dtk_archive}" "${aicc_installer}" "${maccodec_archive}"; rm -rf "${maccodec_tmp}"' EXIT
wget --quiet --tries=5 --timeout=60 --output-document="${maccodec_archive}" "${maccodec_url}"
tar -xzf "${maccodec_archive}" -C "${maccodec_tmp}"
maccodec_env="$(find "${maccodec_tmp}" -type f -path '*/MacCodecSDK/env.sh' -print -quit)"
if [[ -z "${maccodec_env}" ]]; then
  maccodec_env="$(find "${maccodec_tmp}" -type f -name env.sh -print -quit)"
fi
[[ -n "${maccodec_env}" ]] || { echo 'MacCodecSDK env.sh was not found' >&2; exit 1; }
rm -rf /MacCodecSDK
cp -a "$(dirname "${maccodec_env}")" /MacCodecSDK
test -f /MacCodecSDK/env.sh
