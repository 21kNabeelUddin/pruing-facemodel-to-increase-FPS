import cv2
import torch
import time
import numpy as np
import sys
import os
from torchvision import transforms
from facenet_pytorch import MTCNN

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import get_base_model
from models.pruned_model import get_pruned_model

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on {device}")
    print("Controls: 'p' to toggle Pruning, 'q' to Quit")

    # 1. Load Models
    print("Loading models...")
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=False, device=device)
    
    original_model = get_base_model()
    original_model.to(device).eval()

    # Recommended pruning config from sensitivity analysis
    skip_config = {
        'repeat_2': [6, 0, 4, 8, 2, 3, 9], 
        'repeat_3': [0]
    }
    pruned_model = get_pruned_model(skip_config=skip_config)
    pruned_model.to(device).eval()

    # 2. Setup Video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Transform for recognition model input
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    use_pruned = False
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Face Detection
        # Detect face bounding box
        boxes, _ = mtcnn.detect(frame)
        
        has_face = False
        if boxes is not None:
            # Take the first (largest) face
            box = boxes[0].astype(int)
            x1, y1, x2, y2 = box
            
            # Ensure box is within frame boundaries
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                has_face = True

        if not has_face:
            # Fallback to center area if no face detected
            h, w, _ = frame.shape
            size = min(h, w)
            x1, y1 = (w - size) // 2, (h - size) // 2
            x2, y2 = x1 + size, y1 + size
            face_crop = frame[y1:y2, x1:x2]

        # 4. Preprocess & Inference
        input_tensor = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        
        current_model = pruned_model if use_pruned else original_model
        
        start_time = time.time()
        with torch.no_grad():
            _ = current_model(input_tensor)
        end_time = time.time()
        
        # 5. UI Updates
        latency = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        mode_text = "PRUNED" if use_pruned else "ORIGINAL"
        status_color = (0, 255, 0) if use_pruned else (0, 255, 255)
        box_color = (255, 0, 0) if has_face else (50, 50, 50)
        
        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        if not has_face:
             cv2.putText(frame, "NO FACE DETECTED (Center Crop)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(frame, f"Mode: {mode_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Latency: {latency:.1f} ms", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'p' to toggle, 'q' to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Model Optimization Demo", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            use_pruned = not use_pruned
            fps_history = [] 
            print(f"Switched to {'Pruned' if use_pruned else 'Original'} model")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()
