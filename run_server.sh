#!/bin/bash
set -e

echo "Building Cython extensions..."
python3 setup.py build_ext --inplace

echo "Starting TheShow Server..."
python3 theshow.py
