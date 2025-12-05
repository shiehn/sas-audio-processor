#!/bin/bash
# Build script for SAS Audio Processor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect architecture
ARCH=$(uname -m)
echo "Building for architecture: $ARCH"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run PyInstaller
echo "Building binary..."
pyinstaller sas-processor.spec --clean --noconfirm

# Rename output with architecture suffix
OUTPUT_DIR="dist/sas-processor"
if [ -d "$OUTPUT_DIR" ]; then
    # Create architecture-specific name
    NEW_NAME="dist/sas-processor-$ARCH"
    rm -rf "$NEW_NAME"
    mv "$OUTPUT_DIR" "$NEW_NAME"
    echo "Build complete: $NEW_NAME"

    # Show size
    echo ""
    echo "Binary size:"
    du -sh "$NEW_NAME"

    # Test the binary
    echo ""
    echo "Testing binary..."
    "$NEW_NAME/sas-processor" --help
else
    echo "Error: Build output not found"
    exit 1
fi

echo ""
echo "Build successful!"
