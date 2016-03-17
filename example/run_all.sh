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
OUTDIR="${DIR}/output"
echo "Writing output to ${OUTDIR}."
if [ ! -d "${OUTDIR}" ]; then
  mkdir -p "${OUTDIR}"
fi

# Set matplotlib backend
export MPLBACKEND=AGG

# Log what we're doing from now on
set -x

python kalman_filtering.py "${OUTDIR}/kalman_filtering.png"
