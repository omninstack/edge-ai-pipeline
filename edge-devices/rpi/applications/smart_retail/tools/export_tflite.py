import argparse
import tensorflow as tf
import numpy as np
import os

def representative_data_gen():
    """
    Generator function for INT8 quantization.
    Provides representative samples to calibrate the activation ranges.
    In a real scenario, this should load images from your actual dataset.
    """
    for _ in range(100):
        # Create dummy data with shape (1, 640, 640, 3) normalized to [0, 1]
        data = np.random.rand(1, 640, 640, 3).astype(np.float32)
        yield [data]

def export_model(saved_model_dir, output_path, quantize=True):
    print(f"Loading SavedModel from: {saved_model_dir}")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    if quantize:
        print("Enabling INT8 Full Integer Quantization...")
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enforce full integer quantization for all ops
        # This is critical for Edge TPUs, but also great for Pi CPUs
        converter.representative_dataset = representative_data_gen
        
        # Ensure that if any ops can't be quantized, the converter throws an error
        # rather than silently leaving them as float32 (which causes slow context switching)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # Set input/output tensors to uint8 or int8
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    print("Converting model...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Successfully saved TFLite model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to TFLite (INT8)")
    parser.add_argument("--saved_model", required=True, help="Path to input TF SavedModel directory")
    parser.add_argument("--output", required=True, help="Path to save the .tflite model")
    parser.add_argument("--no_quant", action="store_true", help="Disable INT8 quantization (export as FP32)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_model(args.saved_model, args.output, quantize=not args.no_quant)
