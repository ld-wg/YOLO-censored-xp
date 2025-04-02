from ultralytics import YOLO

def main():
    # Instantiate the YOLOv8 model - 'yolov8n.pt' is the lightweight (nano) version.
    # Or 'yolov8s.pt' for the small version if needed.
    model = YOLO('yolov8n.pt')
    
    print("Model loaded successfully:")
    print(model)
    
    # -----------------------------------------
    # Training configuration
    # -----------------------------------------
    # Example data.yaml content:
    #   train: path/to/train/images
    #   val: path/to/val/images
    #   names: ['person', 'car']
    #
    # model.train(data='path/to/data.yaml', epochs=10, imgsz=640)
    
if __name__ == '__main__':
    main()