#!/bin/bash
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.

set -e
MAKE="make --jobs=$NUM_THREADS --keep-going"

if ! $WITH_CUDA; then
  export CPU_ONLY=1
fi
$MAKE all test pycaffe warn lint || true
if ! $WITH_CUDA; then
  $MAKE runtest
fi
$MAKE all
$MAKE test
$MAKE pycaffe
$MAKE pytest
$MAKE warn
if ! $WITH_CUDA; then
  $MAKE lint
fi
