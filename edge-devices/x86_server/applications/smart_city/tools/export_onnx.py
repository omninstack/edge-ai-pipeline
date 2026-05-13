import torch
import torchvision
import argparse
import os

def export_to_onnx(output_path):
    print("Loading pre-trained PyTorch model (ResNet18 as placeholder)...")
    
    # We use a standard pre-trained model for demonstration
    # In a real scenario, you would load your custom state_dict here
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    # Create a dummy input tensor matching the expected input shape
    # Format: (Batch_Size, Channels, Height, Width)
    # We use a batch size of 1 for the trace, but we will define the batch
    # dimension as 'dynamic' below so the server can handle any batch size.
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to ONNX: {output_path}...")
    
    # Export the model
    torch.onnx.export(
        model,                      # model being run
        dummy_input,                # model input (or a tuple for multiple inputs)
        output_path,                # where to save the model (can be a file or file-like object)
        export_params=True,         # store the trained parameter weights inside the model file
        opset_version=14,           # the ONNX version to export the model to
        do_constant_folding=True,   # whether to execute constant folding for optimization
        input_names=['input'],      # the model's input names
        output_names=['output'],    # the model's output names
        
        # Crucial for Edge Servers: Dynamic Axes
        # This allows the ONNX model to accept a variable number of images in a single batch,
        # which is required if you are processing 1 stream one minute, and 10 streams the next.
        dynamic_axes={
            'input': {0: 'batch_size'},    # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print("Successfully exported ONNX model with dynamic batching!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--output", default="../models/smart_city_model.onnx", help="Path to output .onnx file")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_onnx(args.output)
