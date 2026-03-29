#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying dictionary DB to D1 ==="
bash "$SCRIPT_DIR/deploy-db.sh"

echo ""
echo "=== Deploying API Worker ==="
cd "$(dirname "$SCRIPT_DIR")/worker"
npx wrangler deploy

echo ""
echo "=== Publishing trie to consumers ==="
bash "$SCRIPT_DIR/deploy-trie.sh"

echo ""
echo "=== All deployments complete ==="
