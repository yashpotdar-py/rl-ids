#!/usr/bin/env bash
set -eux

dvc push

pushd ../rl-ids-dvc
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git add .
git commit -m "CI: update DVC cache" || echo "No changes"
git push origin main
popd

echo "Pushed to remote successfully"
