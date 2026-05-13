# Intel Edge Defect Inspection Application

This project implements a high-throughput computer vision pipeline optimized for Intel hardware (CPUs, integrated graphics, discrete Arc GPUs) using the **OpenVINO Toolkit**.

## 🚀 Setup & Installation

1. **Environment Setup**: OpenVINO requires standard Python 3.8+.
   ```bash
   cd tools
   pip install -r requirements.txt
   ```

## 🛠️ Model Conversion

OpenVINO works best with its native Intermediate Representation (IR) format (`.xml` network topology and `.bin` weights files). You can convert ONNX, PyTorch, or TensorFlow models to IR using the provided export tool.

```bash
cd tools
python export_openvino.py --onnx ../models/yolov8_defect.onnx --outdir ../models
```

> [!TIP]
> This script automatically compresses weights to FP16, which significantly improves throughput and lowers memory bandwidth usage on Intel processors with minimal impact on accuracy.

## 🏃 Running Inference

Unlike Jetson pipelines which typically execute sequentially to maximize a powerful GPU, OpenVINO is designed to extract maximum performance from multi-core CPUs and iGPUs through asynchronous execution.

This application uses the `ov.AsyncInferQueue` to run multiple inference requests concurrently.

```bash
cd src
python inference.py
```

### Device Selection (`AUTO`)
The inference script uses `device="AUTO"` by default. OpenVINO will automatically detect the best available hardware (e.g., an Intel Core processor's integrated graphics) and seamlessly balance the workload, falling back to the CPU if necessary.
