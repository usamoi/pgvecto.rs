#!/usr/bin/bash
set -e

cd /mnt/build

printf "SEMVER = ${SEMVER}\n"
printf "VERSION = ${VERSION}\n"

export ARCH=$(uname -m)
export PLATFORM=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/amd64/')

apt-get update
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
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
    zlib1g-dev \
    zip
apt-get install -y --no-install-recommends sudo ca-certificates

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
cargo install cargo-pgrx@$(grep 'pgrx = {' Cargo.toml | cut -d '"' -f 2) --debug
cargo pgrx init --pg${VERSION}=/usr/lib/postgresql/${VERSION}/bin/pg_config

cargo build --release --no-default-features --features pg$VERSION
cargo pgrx schema > ./sql/install/vectors--$SEMVER.sql

rm -rf ./build/dir_zip
rm -rf ./build/vectors-pg${VERSION}_${ARCH}-unknown-linux-gnu_${SEMVER}.zip
rm -rf ./build/dir_deb
rm -rf ./build/vectors-pg${VERSION}_${SEMVER}_${PLATFORM}.deb

mkdir -p ./build/dir_zip
cp -a ./sql/upgrade/. ./build/dir_zip/
cp ./sql/install/vectors--$SEMVER.sql ./build/dir_zip/vectors--$SEMVER.sql
cp ./vectors.control ./build/dir_zip/vectors.control
cp ./target/release/libvectors.so ./build/dir_zip/vectors.so
zip vectors-pg${VERSION}_${ARCH}-unknown-linux-gnu_${SEMVER}.zip -j ./build/dir_zip

mkdir -p ./build/dir_deb
mkdir -p ./build/dir_deb/DEBIAN
echo "Package: vectors-pg${VERSION}
Version: ${SEMVER}
Section: database
Priority: optional
Architecture: ${PLATFORM}
Maintainer: Tensorchord <support@tensorchord.ai>
Description: Vector database plugin for Postgres, written in Rust, specifically designed for LLM
Homepage: https://pgvecto.rs/
License: apache2" \
> ./build/dir_deb/DEBIAN/control
for file in ./build/dir_zip/*.sql; do
    F=$(basename "$file")
    cp ./build/dir_deb/$F ./build/dir_deb/usr/share/postgresql/$VERSION/extension/$F
done
for file in ./build/dir_zip/*.control; do
    F=$(basename "$file")
    cp ./build/dir_deb/$F ./build/dir_deb/usr/share/postgresql/$VERSION/extension/$F
done
for file in ./build/dir_zip/*.so; do
    F=$(basename "$file")
    cp ./build/dir_deb/$F ./build/dir_deb/usr/lib/postgresql/$VERSION/lib/$F
done
dpkg --build ./build/dir_deb/ ./build/vectors-pg${VERSION}_${SEMVER}_${PLATFORM}.deb
