import argparse
import openvino as ov
import os

def export_model(onnx_path, output_dir, fp16=True):
    print(f"Loading ONNX model from: {onnx_path}")
    
    # Initialize OpenVINO core
    core = ov.Core()
    
    # Read the model
    model = core.read_model(model=onnx_path)
    
    # Apply FP16 compression if requested
    # Note: FP16 is generally recommended for Intel GPUs and CPUs
    compress_to_fp16 = fp16
    
    # Generate the output base name
    base_name = os.path.basename(onnx_path).split('.')[0]
    if fp16:
        base_name += "_fp16"
        
    output_xml = os.path.join(output_dir, f"{base_name}.xml")
    
    print(f"Converting and optimizing model to OpenVINO IR format...")
    print(f"FP16 compression: {'Enabled' if fp16 else 'Disabled'}")
    
    # Save the model
    ov.save_model(model, output_xml, compress_to_fp16=compress_to_fp16)
    
    print(f"Successfully exported OpenVINO IR model!")
    print(f"XML: {output_xml}")
    print(f"BIN: {output_xml.replace('.xml', '.bin')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX model to OpenVINO IR")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model")
    parser.add_argument("--outdir", required=True, help="Directory to save the IR model (.xml and .bin)")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 compression")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    export_model(args.onnx, args.outdir, fp16=not args.fp32)
