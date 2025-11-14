#!/bin/bash
# Script to build and run the stencil reference generator

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

echo "================================================================"
echo "SHOCCS Stencil Reference Data Generator"
echo "================================================================"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "Configuring CMake..."
echo ""

# Try to configure with common TPL directories
if [ -d "/opt/shoccs-tpl" ]; then
    CMAKE_PREFIX_PATH="/opt/shoccs-tpl"
elif [ -d "$HOME/shoccs-tpl" ]; then
    CMAKE_PREFIX_PATH="$HOME/shoccs-tpl"
else
    CMAKE_PREFIX_PATH=""
fi

if [ -n "$CMAKE_PREFIX_PATH" ]; then
    echo "Using TPL directory: $CMAKE_PREFIX_PATH"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DSHOCCS_TPL_DIR="$CMAKE_PREFIX_PATH" \
          -DBUILD_TESTING=OFF \
          "$PROJECT_ROOT"
else
    echo "No TPL directory found, using system libraries"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=OFF \
          "$PROJECT_ROOT"
fi

echo ""
echo "Building reference generator..."
cmake --build . --target generate_stencil_reference -j$(nproc)

echo ""
echo "Running reference generator..."
./tools/generate_stencil_reference

echo ""
echo "================================================================"
echo "Reference data generated successfully!"
echo "Output: $SCRIPT_DIR/stencil_reference_data.json"
echo "================================================================"
