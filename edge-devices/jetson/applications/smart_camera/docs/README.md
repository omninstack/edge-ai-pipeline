# Jetson Smart Camera Application

This project implements a high-performance computer vision pipeline for NVIDIA Jetson devices (Nano, Xavier, Orin) using TensorRT and PyCUDA.

## 🚀 Setup & Installation

1. **Flash JetPack**: Ensure your Jetson is flashed with NVIDIA JetPack 5.x or 6.x. This provides CUDA, TensorRT, and OpenCV natively.
2. **Install Dependencies**:
   ```bash
   cd tools
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > On Jetson devices, it's often best to use the system-provided `tensorrt` Python wheel located in `/usr/lib/python3.x/dist-packages/`.

## 🛠️ Model Conversion

Before running inference, you must convert an ONNX model into a TensorRT `.engine` file. Engines are specific to the GPU architecture they were compiled on, meaning you **must compile the engine directly on the target Jetson device**.

```bash
cd tools
python export_tensorrt.py --onnx ../models/yolov8n.onnx --engine ../models/smart_camera.engine
```

> [!TIP]
> This script builds an FP16-optimized engine by default, which drastically improves inference speed on Xavier and Orin hardware with minimal accuracy loss.

## 🏃 Running Inference

The inference script grabs frames via OpenCV (easily adaptable to GStreamer for CSI cameras) and executes the highly optimized TensorRT engine.

```bash
cd src
python inference.py
```
