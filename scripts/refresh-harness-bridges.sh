#!/usr/bin/env bash
# Rebuilds the vendored harness bridge bundles in crates/ai/assets/harness
# from a vercel/ai checkout, so they match the commit the Rust harness ports.
#
# Usage: scripts/refresh-harness-bridges.sh <path-to-vercel-ai-checkout>
#
# Needs node and pnpm. The bridges run inside the sandbox under Node, so they
# are vendored as the bundles `tsdown` builds, not ported to Rust.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <path-to-vercel-ai-checkout>" >&2
  exit 1
fi

VERCEL_AI="$(cd "$1" && pwd)"
ASSETS="$(cd "$(dirname "$0")/.." && pwd)/crates/ai/assets/harness"

# Adapter package -> asset directory. Each package's `build` script writes its
# bridge into dist/bridge (index.mjs, package.json, pnpm-lockfile.yaml and,
# where present, host-tool-mcp.mjs and pnpm-workspace.yaml).
BUNDLED=(
  "harness-acp:acp"
  "harness-claude-code:claude-code"
  "harness-codex:codex"
  "harness-opencode:opencode"
)
# Packages whose bridge is only a locked install manifest, read from src/bridge.
MANIFEST_ONLY=(
  "harness-github-copilot:github-copilot"
)

cd "$VERCEL_AI"
echo "vercel/ai at $(git rev-parse --short HEAD)"

filters=(--filter "@vercel/ai-tsdown-config")
for entry in "${BUNDLED[@]}"; do
  filters+=(--filter "@ai-sdk/${entry%%:*}...")
done
pnpm install --frozen-lockfile "${filters[@]}"

for package in provider provider-utils harness; do
  (cd "packages/$package" && pnpm run build)
done

for entry in "${BUNDLED[@]}"; do
  package="${entry%%:*}"
  target="$ASSETS/${entry##*:}"
  (cd "packages/$package" && pnpm run build)
  mkdir -p "$target"
  for file in index.mjs host-tool-mcp.mjs package.json pnpm-lockfile.yaml pnpm-workspace.yaml; do
    if [ -f "packages/$package/dist/bridge/$file" ]; then
      cp "packages/$package/dist/bridge/$file" "$target/$file"
    fi
  done
  echo "refreshed $target"
done

for entry in "${MANIFEST_ONLY[@]}"; do
  package="${entry%%:*}"
  target="$ASSETS/${entry##*:}"
  mkdir -p "$target"
  cp "packages/$package/src/bridge/package.json" "$target/package.json"
  cp "packages/$package/src/bridge/pnpm-lock.yaml" "$target/pnpm-lockfile.yaml"
  cp "packages/$package/src/bridge/pnpm-workspace.yaml" "$target/pnpm-workspace.yaml"
  echo "refreshed $target"
done
