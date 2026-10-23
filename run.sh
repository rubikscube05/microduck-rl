#!/bin/bash
#
# run.sh - watch the most recently trained Microduck policy.
#
cd "$(dirname "$0")"
exec ./.venv/bin/python watch.py "$@"
