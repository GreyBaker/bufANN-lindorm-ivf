#!/usr/bin/env bash
set -euo pipefail
WORKLOAD=insert
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation.sh"
