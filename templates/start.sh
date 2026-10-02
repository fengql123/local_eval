#!/usr/bin/env bash
set -euo pipefail

rm -f /run/eval-ready
touch /run/eval-ready
exec sleep infinity
