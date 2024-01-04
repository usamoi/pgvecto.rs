#!/usr/bin/bash
set -e

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
