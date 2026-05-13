#!/bin/bash
# Edge TPU Compiler Wrapper Script
# This script automates the compilation of a fully quantized INT8 TFLite model
# into an Edge TPU compatible model.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_int8_model.tflite>"
    echo "Example: $0 ../models/traffic_model_int8.tflite"
    exit 1
fi

INPUT_MODEL=$1
OUTPUT_DIR=$(dirname "$INPUT_MODEL")

echo "========================================="
echo " Google Coral Edge TPU Compiler "
echo "========================================="

# Check if the edgetpu_compiler is installed
if ! command -v edgetpu_compiler &> /dev/null
then
    echo "ERROR: edgetpu_compiler could not be found."
    echo "Please install it via: sudo apt-get install edgetpu-compiler"
    exit 1
fi

echo "Compiling model: $INPUT_MODEL"
echo "Output directory: $OUTPUT_DIR"

# Run the compiler
# -s : Show summary
# -a : Attempt to compile all subgraphs
edgetpu_compiler -s -a -o "$OUTPUT_DIR" "$INPUT_MODEL"

echo "========================================="
echo "Compilation Complete."
echo "Your new model should be named: *_edgetpu.tflite"
