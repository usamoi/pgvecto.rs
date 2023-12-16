#!/usr/bin/env bash
set -e

sed -i "s/@CARGO_VERSION@/${SEMVER}/g" ./vectors.control
git config --global user.email "ci@example.com"
git config --global user.name "CI"
git add vectors.control
git commit -m "chore: release"
git tag v$SEMVER
git push origin v$SEMVER
