import cv2
import numpy as np
import time

# We import tflite_runtime instead of tensorflow to save massive amounts of RAM and disk space on the Pi
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("WARNING: tflite_runtime not found. Falling back to full tensorflow for testing.")
    import tensorflow.lite as tflite

class TFLiteInference:
    def __init__(self, model_path, num_threads=4):
        print(f"Loading TFLite model from {model_path}...")
        
        # Load the TFLite model and allocate tensors.
        # We explicitly enable the XNNPACK delegate, which provides highly optimized
        # assembly routines for ARM CPUs (like the Cortex-A72/A76 in the Pi 4/5).
        try:
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_delegates=[tflite.load_delegate('libxnnpack.so')]
            )
            print("Successfully loaded XNNPACK delegate.")
        except Exception as e:
            print(f"Could not load XNNPACK delegate, falling back to standard CPU ops. Error: {e}")
            self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
            
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Check if the model requires quantized (uint8/int8) or float32 input
        self.input_dtype = self.input_details[0]['dtype']
        self.input_scale, self.input_zero_point = self.input_details[0]['quantization']
        
        # Target size from model metadata
        self.target_height = self.input_details[0]['shape'][1]
        self.target_width = self.input_details[0]['shape'][2]

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.target_width, self.target_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.input_dtype == np.uint8:
            # If the model is INT8/UINT8 quantized, pass pixels directly
            img = np.expand_dims(img, axis=0)
            return img
        else:
            # If FP32, normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img

    def infer(self, frame):
        input_data = self.preprocess(frame)
        
        # Set tensor and invoke
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        start_time = time.time()
        self.interpreter.invoke()
        latency = (time.time() - start_time) * 1000
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data, latency

def main():
    model_path = "../models/smart_retail_int8.tflite"
    
    try:
        # Use 4 threads to maximize utilization of the Pi's quad-core CPU
        infer_engine = TFLiteInference(model_path, num_threads=4)
    except Exception as e:
        print(f"Failed to load model. Did you run export_tflite.py? Error: {e}")
        return

    # Open USB webcam or Pi Camera Module
    cap = cv2.VideoCapture(0)
    
    print("Starting TFLite inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        output, latency = infer_engine.infer(frame)
        
        print(f"Inference Latency: {latency:.2f} ms")
        
        # Placeholder: Parse output (e.g., SSD bounding boxes) and publish to MQTT
        
    cap.release()

if __name__ == "__main__":
    main()
