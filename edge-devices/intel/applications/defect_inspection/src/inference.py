import cv2
import numpy as np
import openvino as ov
import time

class AsyncOpenVINOInfer:
    def __init__(self, model_xml, device="AUTO", num_requests=4):
        self.core = ov.Core()
        print(f"Loading model: {model_xml}")
        print(f"Target Device: {device}")
        
        # Read and compile the model
        model = self.core.read_model(model=model_xml)
        
        # We can enable performance hints for throughput to maximize FPS
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = self.core.compile_model(model, device_name=device, config=config)
        
        # Setup Async Infer Queue
        self.infer_queue = ov.AsyncInferQueue(self.compiled_model, jobs=num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Get expected shape (N, C, H, W)
        self.shape = self.input_layer.shape
        self.target_size = (self.shape[3], self.shape[2]) # (W, H)
        
        self.frame_count = 0
        self.start_time = time.time()

    def completion_callback(self, request, userdata):
        """Callback executed when an async inference request completes."""
        frame_id, original_frame = userdata
        
        # Extract output tensor
        output_tensor = request.get_output_tensor(self.output_layer.index).data
        
        # Post-processing (Placeholder for defect inspection logic)
        # e.g., output_tensor might be a segmentation mask or bounding boxes
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            fps = self.frame_count / (time.time() - self.start_time)
            print(f"Async Throughput: {fps:.2f} FPS")

    def preprocess(self, frame):
        """Resize, transpose, and expand dims for OpenVINO."""
        img = cv2.resize(frame, self.target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def enqueue(self, frame, frame_id):
        """Push a frame to the async queue."""
        input_tensor = self.preprocess(frame)
        self.infer_queue.start_async({self.input_layer.any_name: input_tensor}, userdata=(frame_id, frame))

    def wait_all(self):
        """Wait for all pending requests to finish."""
        self.infer_queue.wait_all()

def main():
    # Example paths
    model_path = "../models/defect_model_fp16.xml"
    
    try:
        # Initializing the engine (defaults to AUTO, letting OpenVINO pick CPU or iGPU)
        # Using 4 async requests for overlapping execution
        engine = AsyncOpenVINOInfer(model_path, device="AUTO", num_requests=4)
    except Exception as e:
        print(f"Failed to load OpenVINO model. Did you run export_openvino.py? Error: {e}")
        return

    # Open standard camera (or video stream of a conveyor belt)
    cap = cv2.VideoCapture(0)
    
    print("Starting asynchronous inference loop...")
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Push the frame to the infer queue
        # If the queue is full, this call blocks until a slot is free, naturally pacing the loop
        engine.enqueue(frame, frame_id)
        frame_id += 1
        
        # Display logic could go here, but beware cv2.imshow blocks
        
    # Drain the queue before exiting
    engine.wait_all()
    cap.release()

if __name__ == "__main__":
    main()
