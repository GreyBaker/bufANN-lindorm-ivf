#!/usr/bin/env bash
set -euo pipefail
WORKLOAD=delete
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_mutation_prepare.sh"
