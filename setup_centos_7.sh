#!/usr/bin/env bash
set -euo pipefail

# ---- Versions (override by env if you want) ----
CMAKE_VERSION="${CMAKE_VERSION:-3.30.4}"       # current latest shown on cmake.org downloads :contentReference[oaicite:2]{index=2}
OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.30}" # latest tag in OpenBLAS releases :contentReference[oaicite:3]{index=3}

# Where to install
PREFIX_CMAKE="/usr/local/cmake-${CMAKE_VERSION}"
PREFIX_OPENBLAS="/usr/local/openblas-${OPENBLAS_VERSION}"

# CentOS 7 is EOL; yum often needs vault repos. Keep it minimal.
VAULT_VER="${VAULT_VER:-7.9.2009}"

require_root() { [[ $EUID -eq 0 ]] || { echo "Run as root (sudo bash $0)"; exit 1; }; }

log() { echo -e "\n[setup] $*\n"; }

fix_centos7_repos_to_vault() {
  log "Pointing CentOS 7 repos to vault.centos.org (best-effort)"
  for f in /etc/yum.repos.d/CentOS-*.repo; do
    [[ -f "$f" ]] || continue
    sed -i \
      -e 's|^mirrorlist=|#mirrorlist=|g' \
      -e 's|^#baseurl=|baseurl=|g' \
      -e "s|mirror.centos.org/centos/\\\$releasever|vault.centos.org/${VAULT_VER}|g" \
      -e "s|mirror.centos.org/centos/7|vault.centos.org/${VAULT_VER}|g" \
      "$f" || true
  done
  yum clean all -q || true
  yum makecache -q || true
}

install_prereqs() {
  log "Installing build prerequisites"
  yum -y install \
    yum-utils curl wget git which tar gzip bzip2 xz \
    make automake autoconf libtool patch \
    perl python3 \
    openssl-devel \
    >/dev/null
}

install_devtoolset11() {
  log "Installing devtoolset-11 toolchain (GCC/G++)"
  yum -y install centos-release-scl centos-release-scl-rh >/dev/null || true
  yum -y install devtoolset-11-toolchain devtoolset-11-gcc-gfortran >/dev/null

  # Make it available in new shells
  cat >/etc/profile.d/enable-devtoolset-11.sh <<'EOF'
if [[ -f /opt/rh/devtoolset-11/enable ]]; then
  source /opt/rh/devtoolset-11/enable
fi
EOF
  chmod 0644 /etc/profile.d/enable-devtoolset-11.sh

  # Enable for THIS script run
  source /opt/rh/devtoolset-11/enable
}

build_and_install_cmake() {
  log "Building CMake ${CMAKE_VERSION} from source into ${PREFIX_CMAKE}"

  mkdir -p /usr/local/src
  cd /usr/local/src

  local tar="cmake-${CMAKE_VERSION}.tar.gz"
  local url="https://cmake.org/files/v${CMAKE_VERSION%.*}/${tar}"

  if [[ ! -f "$tar" ]]; then
    wget -q "$url" -O "$tar"
  fi

  rm -rf "cmake-${CMAKE_VERSION}"
  tar -xf "$tar"
  cd "cmake-${CMAKE_VERSION}"

  ./bootstrap --prefix="${PREFIX_CMAKE}"
  make -j"$(nproc)"
  make install

  # Convenience symlinks
  ln -sf "${PREFIX_CMAKE}/bin/cmake" /usr/local/bin/cmake
  ln -sf "${PREFIX_CMAKE}/bin/ctest" /usr/local/bin/ctest
  ln -sf "${PREFIX_CMAKE}/bin/cpack" /usr/local/bin/cpack

  cmake --version
}

build_and_install_openblas() {
  log "Building OpenBLAS ${OPENBLAS_VERSION} from source into ${PREFIX_OPENBLAS}"

  mkdir -p /usr/local/src
  cd /usr/local/src

  local tag="v${OPENBLAS_VERSION}"
  local tar="OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
  local url="https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/${tag}.tar.gz"

  if [[ ! -f "$tar" ]]; then
    wget -q "$url" -O "$tar"
  fi

  rm -rf "OpenBLAS-${OPENBLAS_VERSION}"
  tar -xf "$tar"
  cd "OpenBLAS-${OPENBLAS_VERSION}"

  # Common settings:
  # - DYNAMIC_ARCH=1 builds multiple kernels for different CPUs
  # - USE_OPENMP=1 enables OpenMP parallelism (works well for many workloads)
  make -j"$(nproc)" DYNAMIC_ARCH=1 USE_OPENMP=1
  make PREFIX="${PREFIX_OPENBLAS}" install

  # Make runtime linker find it
  echo "${PREFIX_OPENBLAS}/lib" >/etc/ld.so.conf.d/openblas.conf
  ldconfig

  # Convenience: stable symlink
  ln -sfn "${PREFIX_OPENBLAS}" /usr/local/openblas

  # Quick sanity
  ls -l /usr/local/openblas/lib | head -n 20 || true
}

main() {
  require_root
  fix_centos7_repos_to_vault
  install_prereqs
  install_devtoolset11
  build_and_install_cmake
  build_and_install_openblas

  log "Done."
  echo "Open a new shell (or run: source /etc/profile) then verify:"
  echo "  g++ --version"
  echo "  cmake --version"
  echo "  ldconfig -p | grep -i openblas"
  echo
  echo "Notes:"
  echo "  - CMake installed at: ${PREFIX_CMAKE} (symlinked to /usr/local/bin/cmake)"
  echo "  - OpenBLAS installed at: ${PREFIX_OPENBLAS} (symlinked to /usr/local/openblas)"
}

main "$@"
