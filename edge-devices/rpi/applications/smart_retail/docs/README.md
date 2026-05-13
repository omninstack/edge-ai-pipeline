# Raspberry Pi Smart Retail Application

This project implements a lightweight computer vision pipeline designed to run efficiently on the Arm CPUs found in Raspberry Pi devices (Pi 4, Pi 5, etc.).

## 🚀 Setup & Installation

Raspberry Pis have limited RAM and storage. We highly recommend using `tflite-runtime` instead of the full `tensorflow` library for inference deployments.

```bash
cd tools
pip install -r requirements.txt
```

> [!NOTE]
> To run the `export_tflite.py` conversion script, you must run it on your powerful Host PC (Mac/Windows/Linux) which has the full `tensorflow` package installed.

## 🛠️ Model Conversion & Quantization

To get real-time frames-per-second on a Pi CPU, you **must** quantize your model to INT8 (8-bit integers). Floating-point (FP32) math is extremely slow on Arm CPUs without an accelerator.

Run this on your Host PC:
```bash
cd tools
python export_tflite.py --saved_model ../models/my_tf_model/ --output ../models/smart_retail_int8.tflite
```

This script enforces Full Integer Quantization.

## 🏃 Running Inference

Deploy the `smart_retail` folder to your Raspberry Pi. The inference script automatically utilizes the **XNNPACK delegate**, which is a highly optimized library of neural network assembly routines for Arm processors.

```bash
cd src
python inference.py
```

### Extending with Edge TPU (Google Coral)
If you add a Google Coral USB Accelerator to your Pi, you can easily modify `inference.py` to use the `libedgetpu.so.1` delegate instead of `libxnnpack.so`. This will offload the INT8 math to the TPU, accelerating inference by up to 10x!
