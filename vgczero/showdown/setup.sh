#!/usr/bin/env bash
# Clone and build the pinned upstream Pokemon Showdown that vgczero treats as
# the source of truth for dex data, stat calculation and parity tests.
# Usage: vgczero/showdown/setup.sh   (or set PS_DIR to an existing build)
set -euo pipefail
PS_SHA=9fb3a5b99f1a0bea17f495c5cc1bfe04fdd19c3e   # smogon/pokemon-showdown master, 2026-10-03
HERE="$(cd "$(dirname "$0")" && pwd)"
PS_DIR="${PS_DIR:-$HERE/ps}"
if [ ! -d "$PS_DIR/.git" ]; then
  git clone https://github.com/smogon/pokemon-showdown.git "$PS_DIR"
fi
cd "$PS_DIR"
git fetch --depth 1 origin "$PS_SHA" 2>/dev/null || true
git checkout -q "$PS_SHA"
npm install --no-audit --no-fund
node build
echo "Showdown ready at $PS_DIR"
