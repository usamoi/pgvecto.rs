#!/usr/bin/bash
set -e

cd /mnt/build

printf "SEMVER = ${SEMVER}\n"
printf "VERSION = ${VERSION}\n"
printf "ARCH = ${ARCH}\n"
printf "PLATFORM = ${PLATFORM}\n"

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
apt-get update
apt-get install -y sudo
sudo --preserve-env apt-get install -y \
    bison \
    build-essential \
    ccache \
    curl \
    flex \
    gcc \
    git \
    gnupg \
    libreadline-dev \
    libssl-dev \
    libxml2-dev \
    libxml2-utils \
    libxslt-dev \
    lsb-release \
    pkg-config \
    tzdata \
    wget \
    xsltproc \
    zlib1g-dev

# 3
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | sudo tee -a /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y --no-install-recommends postgresql-${VERSION} postgresql-server-dev-${VERSION}

# 4
sudo sh -c 'echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-16 main" >> /etc/apt/sources.list'
wget --quiet -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y --no-install-recommends clang-16

# 5
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
source "$HOME/.cargo/env"

# 6
cargo install cargo-pgrx@$(grep 'pgrx = {' Cargo.toml | cut -d '"' -f 2)
cargo pgrx init --pg${VERSION}=/usr/lib/postgresql/${VERSION}/bin/pg_config

# 7
sudo chmod 777 /usr/share/postgresql/${VERSION}/extension/
sudo chmod 777 /usr/lib/postgresql/${VERSION}/lib/

sudo apt-get -y install ruby-dev libarchive-tools
sudo gem install --no-document fpm

mkdir -p ./build/vectors-pg${VERSION}_${SEMVER}_${PLATFORM}
./scripts/package.sh
mv ./target/release/vectors-pg${VERSION}/usr ./build/vectors-pg${VERSION}_${SEMVER}_${PLATFORM}/usr

(cd ./build/vectors-pg${VERSION}_${SEMVER}_${PLATFORM} && fpm \
  --input-type dir \
  --output-type deb \
  --name vectors-pg${VERSION} \
  --version ${SEMVER} \
  --license apache2 \
  --deb-no-default-config-files \
  --package ../vectors-pg${VERSION}_${SEMVER}_${PLATFORM}.deb \
  --architecture ${PLATFORM} \
  .)
