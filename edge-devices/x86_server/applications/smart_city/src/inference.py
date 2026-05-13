import cv2
import numpy as np
import onnxruntime as ort
import time

class ONNXInference:
    def __init__(self, model_path):
        print(f"Loading ONNX model from {model_path}...")
        
        # ONNX Runtime Execution Providers (EPs)
        # The runtime will attempt to load these in the order provided.
        # If TensorRT is not available, it falls back to CUDA. 
        # If no Nvidia GPU is found, it falls back to the CPU EP.
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': True,
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }),
            'CPUExecutionProvider',
        ]
        
        # SessionOptions allow tuning thread behavior for CPU execution
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Threads for a single op
        sess_options.inter_op_num_threads = 2  # Threads for executing ops in parallel
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        active_providers = self.session.get_providers()
        print(f"Active Execution Provider: {active_providers[0]}")
        
        # Get input metadata
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.target_height = input_shape[2] if isinstance(input_shape[2], int) else 224
        self.target_width = input_shape[3] if isinstance(input_shape[3], int) else 224

    def preprocess(self, frames):
        """
        Preprocesses a list of frames to create a batch tensor.
        This demonstrates dynamic batching for multi-camera server environments.
        """
        batch = []
        for frame in frames:
            img = cv2.resize(frame, (self.target_width, self.target_height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            # ImageNet normalization
            img -= np.array([0.485, 0.456, 0.406])
            img /= np.array([0.229, 0.224, 0.225])
            # HWC to CHW
            img = np.transpose(img, (2, 0, 1))
            batch.append(img)
            
        # Stack into (Batch, C, H, W)
        return np.stack(batch, axis=0)

    def infer(self, frames):
        input_tensor = self.preprocess(frames)
        
        start_time = time.time()
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        latency = (time.time() - start_time) * 1000
        
        return outputs[0], latency

def main():
    model_path = "../models/smart_city_model.onnx"
    
    try:
        infer_engine = ONNXInference(model_path)
    except Exception as e:
        print(f"Failed to load ONNX model. Did you run export_onnx.py? Error: {e}")
        return

    # Simulate multi-camera streaming by grabbing multiple frames
    # In a real server, these would come from RTSP streams via threads
    cap = cv2.VideoCapture(0)
    
    print("Starting Multi-Camera ONNX inference loop...")
    while True:
        # Grab a "batch" of 4 identical frames just to simulate 4 cameras
        ret, frame = cap.read()
        if not ret:
            break
            
        simulated_cameras = [frame, frame, frame, frame]
            
        outputs, latency = infer_engine.infer(simulated_cameras)
        
        # Calculate per-frame latency
        per_frame_latency = latency / len(simulated_cameras)
        
        print(f"Batch Latency: {latency:.2f} ms | Per-Camera Latency: {per_frame_latency:.2f} ms | Batch Size: {len(simulated_cameras)}")
        
        # Optional: Post-process outputs (e.g. Argmax for classification)
        
    cap.release()

if __name__ == "__main__":
    main()
