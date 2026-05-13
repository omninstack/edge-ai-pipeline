import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
    def load_engine(self):
        print(f"Loading engine from {self.engine_path}...")
        with open(self.engine_path, 'rb') as f:
            return self.runtime.deserialize_cuda_engine(f.read())
            
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings, stream

    def infer(self, image):
        # 1. Copy image to host input buffer
        np.copyto(self.inputs[0]['host'], image.ravel())
        
        # 2. Transfer input data to the GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 3. Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 4. Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host']

def main():
    engine_path = "../models/smart_camera.engine"
    
    # Initialize TensorRT engine
    try:
        trt_infer = TRTInference(engine_path)
    except Exception as e:
        print(f"Failed to load engine (make sure you built it with export_tensorrt.py): {e}")
        return

    # Use GStreamer pipeline for Jetson optimized capture (placeholder)
    # cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    
    print("Starting inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess (resize to expected model input, e.g., 640x640)
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        start_time = time.time()
        output = trt_infer.infer(img)
        latency = (time.time() - start_time) * 1000
        
        print(f"Inference Latency: {latency:.2f} ms")
        
        # Placeholder for post-processing and MQTT publishing
        # ...
        
    cap.release()

if __name__ == "__main__":
    main()
