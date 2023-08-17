#!/usr/bin/bash

apt-get update
apt-get install -y wget
apt-get install -y curl
apt-get install -y lsb-release
apt-get install -y gnupg
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
apt-get update
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install tzdata
apt-get install -y build-essential
apt-get install -y libpq-dev
apt-get install -y libssl-dev
apt-get install -y pkg-config
apt-get install -y gcc
apt-get install -y libreadline-dev
apt-get install -y flex
apt-get install -y bison
apt-get install -y libxml2-dev
apt-get install -y libxslt-dev
apt-get install -y libxml2-utils
apt-get install -y xsltproc
apt-get install -y zlib1g-dev
apt-get install -y ccache
apt-get install -y clang
apt-get install -y git
apt-get install -y postgresql-15
apt-get install -y postgresql-server-dev-15
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rev=$(cat Cargo.toml | grep "pgrx =" | awk -F 'rev = "' '{print $2}' | cut -d'"' -f1)
cargo install cargo-pgrx --git https://github.com/tensorchord/pgrx.git --rev $rev
cargo pgrx init --pg15=/usr/lib/postgresql/15/bin/pg_config
cargo pgrx install --release
