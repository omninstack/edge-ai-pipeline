import cv2
import numpy as np
import time
import argparse

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import detect

class EdgeTPUInference:
    def __init__(self, model_path):
        print(f"Loading Edge TPU model from {model_path}...")
        
        try:
            # make_interpreter automatically finds the Edge TPU (USB or PCIe)
            # and loads the libedgetpu delegate.
            self.interpreter = make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            print("Successfully initialized Edge TPU!")
        except Exception as e:
            print(f"Failed to initialize Edge TPU. Is it plugged in? Error: {e}")
            raise

    def infer(self, frame):
        # Resize the image using the helper from PyCoral adapters
        # which pulls the exact expected shape from the TFLite metadata
        _, scale = common.set_resized_input(
            self.interpreter, frame.shape[:2],
            lambda size: cv2.resize(frame, size)
        )
        
        start_time = time.time()
        self.interpreter.invoke()
        latency = (time.time() - start_time) * 1000
        
        # Extract predictions (e.g., bounding boxes for SSD/YOLO)
        # This is a generic object detection parsing provided by PyCoral
        # We use a threshold of 0.4 (40% confidence)
        objs = detect.get_objects(self.interpreter, score_threshold=0.4, image_scale=scale)
        
        return objs, latency

def main():
    parser = argparse.ArgumentParser(description="Edge TPU Traffic Monitoring Inference")
    parser.add_argument("--model", default="../models/traffic_model_int8_edgetpu.tflite", help="Path to *_edgetpu.tflite model")
    args = parser.parse_args()
    
    try:
        infer_engine = EdgeTPUInference(args.model)
    except Exception:
        return

    # Open standard camera
    cap = cv2.VideoCapture(0)
    
    print("Starting Edge TPU inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) to RGB (TFLite standard)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        objs, latency = infer_engine.infer(rgb_frame)
        
        print(f"Inference Latency: {latency:.2f} ms | Objects Detected: {len(objs)}")
        
        # Optional: Draw bounding boxes and publish to MQTT here
        
    cap.release()

if __name__ == "__main__":
    main()
