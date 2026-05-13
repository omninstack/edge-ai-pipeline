import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, fp16_mode=True):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building with FP16 optimization...")
        
    print(f"Parsing ONNX file {onnx_file_path}...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    print(f"Building TensorRT engine. This may take a while...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Failed to build engine.")
        return None
        
    with open(engine_file_path, 'wb') as f:
        f.write(engine_bytes)
    print(f"Successfully saved TensorRT engine to {engine_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model")
    parser.add_argument("--engine", required=True, help="Path to output TRT engine")
    parser.add_argument("--fp32", action="store_true", help="Disable FP16 and build FP32 engine")
    args = parser.parse_args()
    
    build_engine(args.onnx, args.engine, fp16_mode=not args.fp32)
