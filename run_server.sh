#!/bin/bash
set -e

# Define environment directory
VENV_DIR="venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Build Cython extensions
echo "Building Cython extensions..."
python3 setup.py build_ext --inplace

# Run server
echo "Starting The Show..."
python3 theshow.py
