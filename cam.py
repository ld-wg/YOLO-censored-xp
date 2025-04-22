#!/usr/bin/env python3
# Webcam inference script for YOLOv8 trained models
# Runs object detection on webcam

import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on webcam")
    parser.add_argument('--model', type=str, default='runs/train/uncensored_frac_0_01_10ep_/weights/best.pt',
                       help='Path to YOLOv8 model weights (PT file)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (mps, cuda, cpu). Default: auto-detect')
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_args()
    
    # determine best device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        else:
            device = "cpu"  # CPU only
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # load YOLOv8 model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Available models:")
        for path in Path("runs/train").glob("*/weights/best.pt"):
            print(f"  {path}")
        return
    
    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    model.to(device)
    
    # open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # run detection on frame
        results = model(frame, conf=args.conf, device=device)
        
        # draw bounding boxes on frame
        annotated_frame = results[0].plot()
        
        # display result
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
        
        # check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main() 