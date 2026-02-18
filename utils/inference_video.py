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
from models.pruned_model import get_pruned_model

def run_video_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running video inference on {device}")
    
    # 1. Setup Models
    # MTCNN handles DETECTION, InceptionResnet handles RECOGNITION (Features)
    # 1. Setup Models
    # MTCNN handles DETECTION, InceptionResnet handles RECOGNITION (Features)
    # Tuning for surveillance: Lower thresholds and smaller min_face_size
    mtcnn = MTCNN(
        keep_all=True, 
        device=device,
        min_face_size=50, # Increased from 15 to remove small noise
        thresholds=[0.7, 0.8, 0.8] # Stricter thresholds to remove false positives
    ) 
    
    skip_config = {
        'repeat_1': [int(x) for x in args.prune_repeat_1.split(',')] if args.prune_repeat_1 else [],
        'repeat_2': [int(x) for x in args.prune_repeat_2.split(',')] if args.prune_repeat_2 else [],
        'repeat_3': [int(x) for x in args.prune_repeat_3.split(',')] if args.prune_repeat_3 else []
    }
    
    model = get_pruned_model(skip_config=skip_config)
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    # 2. Setup Video
    video_path = args.input
    if not os.path.exists(video_path):
        # Check project root
        root_path = os.path.join(os.path.dirname(__file__), '..', args.input)
        if os.path.exists(root_path):
            video_path = root_path
        else:
            print(f"Error: Video file {args.input} not found in current or root directory.")
            return

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup Output
    output_path = "inference_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Processing video: {args.input}...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detection
        boxes, _ = mtcnn.detect(frame)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                # Clamp to frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Inference (Recognition Engine)
                    input_tensor = transform(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                    
                    start_t = time.time()
                    with torch.no_grad():
                        _ = model(input_tensor) # Feature extraction
                    latency = (time.time() - start_t) * 1000
                    
                    # Draw UI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Pruned Model Latency: {latency:.1f}ms", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='emotion_happy.mp4')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pruned_epoch_10.pth')
    parser.add_argument('--prune_repeat_1', type=str, default='')
    parser.add_argument('--prune_repeat_2', type=str, default='6,0,4,8,2,3,9')
    parser.add_argument('--prune_repeat_3', type=str, default='0')
    
    args = parser.parse_args()
    run_video_inference(args)
