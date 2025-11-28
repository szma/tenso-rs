#!/bin/bash
# Download MNIST dataset

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    curl -O "$BASE_URL/$file"
    echo "Extracting $file..."
    gunzip -f "$file"
done

echo "Done! Files:"
ls -la
