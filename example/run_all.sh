#!/bin/bash
#
# Script to run all examples saving the output to a temporary directory.

echo "Running all examples..."

# Exit on error
set -e

# Change to this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${DIR}"

# Create temporary directory for output
OUTDIR=$(mktemp "${DIR}/example-output-$(date -I)-XXXXXX")
echo "Writing output to ${OUTDIR}."

# Log what we're doing from now on
set -x

python kalman_filtering.py "${OUTDIR}/kalman_filtering.png"
