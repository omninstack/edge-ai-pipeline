# Google Coral Edge TPU Traffic Monitoring

This project implements a highly accelerated computer vision pipeline designed for the Google Coral Dev Board or the Coral USB Accelerator.

## 🚀 Setup & Installation

The Edge TPU requires specific system-level drivers to function. 
For Debian/Ubuntu/Raspberry Pi OS systems, we highly recommend using `apt` rather than `pip`.

```bash
# 1. Add the Coral Debian repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

# 2. Install the Edge TPU Runtime and PyCoral API
sudo apt-get install libedgetpu1-std python3-pycoral
```
> [!TIP]
> You can install `libedgetpu1-max` instead of `libedgetpu1-std` to run the TPU at maximum clock speed (but it will get significantly hotter!).

## 🛠️ Model Compilation

The Edge TPU is an ASIC that only understands specific, low-level instructions. An INT8 quantized `.tflite` model cannot run on it directly. You must compile the model using the `edgetpu_compiler`.

1. Install the compiler on your Host PC (Linux only):
   ```bash
   sudo apt-get install edgetpu-compiler
   ```
2. Run the provided script on your INT8 model:
   ```bash
   cd tools
   bash compile_edgetpu.sh ../models/my_model_int8.tflite
   ```

The script will output a new file named `my_model_int8_edgetpu.tflite`. This is the file you deploy to the edge device.

## 🏃 Running Inference

We use the official `pycoral` API which abstracts away much of the boilerplate required to map operations and load delegates.

```bash
cd src
python inference.py --model ../models/my_model_int8_edgetpu.tflite
```

### CPU Fallback (Co-compilation)
If the compiler encounters a layer that the Edge TPU ASIC cannot execute, it will map that specific layer back to the host CPU. `pycoral` handles this data-transfer seamlessly, though excessive CPU fallback will drastically increase your latency. Always check the compiler output summary!
