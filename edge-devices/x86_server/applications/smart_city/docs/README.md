# x86 Edge Server (ONNX Runtime) Application

This project implements a high-throughput, multi-batch computer vision pipeline designed for edge servers, local network racks, and beefy x86 edge nodes. It leverages **ONNX Runtime** (`onnxruntime-gpu`), the industry standard for scalable server deployments.

## 🚀 Setup & Installation

The edge server environment relies on `onnxruntime-gpu`, which provides access to Nvidia hardware acceleration but gracefully falls back to CPU execution if no GPU is present.

```bash
cd tools
pip install -r requirements.txt
```

> [!NOTE]
> If you are running this on a server with an Nvidia GPU, ensure you have the appropriate CUDA Toolkit and cuDNN libraries installed on the host OS for `onnxruntime-gpu` to function correctly.

## 🛠️ Model Export

In this pipeline, models are designed to be trained in PyTorch and exported to ONNX. The critical feature of this export process is the use of **Dynamic Axes**.

```bash
cd tools
python export_onnx.py --output ../models/smart_city_model.onnx
```

By defining the batch size as a dynamic axis during the export, the `.onnx` file can accept 1 image, 4 images, or 32 images in a single inference call. This is vital for servers ingesting a variable number of RTSP streams from city cameras.

## 🏃 Running Inference

The inference script (`src/inference.py`) demonstrates two key server concepts:

1. **Execution Provider Fallback**: The code requests the `TensorrtExecutionProvider` first. If TensorRT libraries aren't found, it requests the `CUDAExecutionProvider`. If no Nvidia GPU exists, it falls back to the `CPUExecutionProvider`.
2. **Dynamic Batching**: The script simulates pulling frames from multiple cameras and stacking them into a single Numpy tensor before passing them to the ONNX session.

```bash
cd src
python inference.py
```
