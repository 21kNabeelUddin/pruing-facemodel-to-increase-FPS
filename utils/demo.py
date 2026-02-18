import cv2
import torch
import time
import numpy as np
import sys
import os
import argparse
from torchvision import transforms
from facenet_pytorch import MTCNN

# Ensure we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.base_model import get_base_model
from models.pruned_model import get_pruned_model

def draw_text_with_shadow(img, text, pos, font, scale, color, thickness, shadow_color=(0, 0, 0)):
    """Draws text with a slight shadow for better visibility on complex backgrounds."""
    x, y = pos
    # Draw shadow
    cv2.putText(img, text, (x+2, y+2), font, scale, shadow_color, thickness + 1)
    # Draw main text
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running demo on {device}")
    print("Controls: 'p' to toggle Pruning, 'q' to Quit")

    # 1. Load Models
    print("Loading models...")
    # Initialize MTCNN for face detection
    # Tuned to reduce false positives (stricter thresholds, larger min face)
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=60, thresholds=[0.7, 0.8, 0.8])
    
    original_model = get_base_model()
    original_model.to(device).eval()

    # Determine which layers to skip for the pruned model
    skip_config = {}
    if args.prune_repeat_1:
         skip_config['repeat_1'] = [int(x) for x in args.prune_repeat_1.split(',')]
    if args.prune_repeat_2:
        skip_config['repeat_2'] = [int(x) for x in args.prune_repeat_2.split(',')]
    if args.prune_repeat_3:
         skip_config['repeat_3'] = [int(x) for x in args.prune_repeat_3.split(',')]

    pruned_model = get_pruned_model(skip_config=skip_config)
    
    if os.path.exists(args.checkpoint):
        print(f"Loading trained weights from {args.checkpoint}...")
        pruned_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with un-tuned pruned model.")
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
        # Detect face bounding box (MTCNN)
        boxes, _ = mtcnn.detect(frame)
        
        has_face = False
        if boxes is not None:
            # Take the largest face
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
            # Fallback to center area
            h, w, _ = frame.shape
            size = min(h, w)
            x1, y1 = (w - size) // 2, (h - size) // 2
            x2, y2 = x1 + size, y1 + size
            face_crop = frame[y1:y2, x1:x2]

        # 4. Preprocess & Inference
        input_tensor = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        
        current_model = pruned_model if use_pruned else original_model
        
        # PERFORMANCE CALCULATION:
        # We only measure the time spent inside the Recognition Model (InceptionResnet).
        # We do NOT include preprocessing or MTCNN detection in this measurement
        # because those parts are identical for both "Original" and "Pruned" modes.
        # This gives a "pure" comparison of the model engine speed.
        
        start_time = time.time()
        with torch.no_grad():
            _ = current_model(input_tensor)
        end_time = time.time()
        
        # 5. UI Updates
        # Latency = time in milliseconds
        # FPS = 1.0 / time in seconds
        latency = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)
        
        fps_history.append(fps)
        if len(fps_history) > 30: # 30-frame moving average for stability
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        mode_text = "PRUNED" if use_pruned else "ORIGINAL"
        status_color = (0, 255, 0) if use_pruned else (0, 255, 255) # Green vs Yellow
        box_color = (255, 0, 0) if has_face else (50, 50, 50)
        
        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        if not has_face:
             draw_text_with_shadow(frame, "NO FACE (Center Crop)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # High visibility UI
        draw_text_with_shadow(frame, f"Mode: {mode_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        draw_text_with_shadow(frame, f"FPS: {avg_fps:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2) # Cyan/Yellow text
        draw_text_with_shadow(frame, f"Latency: {latency:.1f} ms", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2) # Cyan/Yellow text
        
        draw_text_with_shadow(frame, "Press 'p' to toggle, 'q' to quit", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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
    # Get the project root directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    default_checkpoint = os.path.join(project_root, 'checkpoints', 'pruned_epoch_10.pth')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint)
    parser.add_argument('--prune_repeat_1', type=str, default='')
    parser.add_argument('--prune_repeat_2', type=str, default='6,0,4,8,2,3,9') # Default to sensitivity report findings
    parser.add_argument('--prune_repeat_3', type=str, default='0')
    
    args = parser.parse_args()
    run_demo(args)
