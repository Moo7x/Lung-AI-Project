from ultralytics import YOLO
import os

def main():
    print("Loading YOLOv8-Medium model...")
    model = YOLO('yolov8m.pt') 

    yaml_path = os.path.join(os.getcwd(), 'datasets', 'lung_data', 'data.yaml')

    print(f"Training on data at: {yaml_path}")
    results = model.train(
        data=yaml_path,
        
        # --- SPEED SETTINGS ---
        epochs=30,      
        imgsz=640,
        
        # --- GPU SETTINGS (CRITICAL) ---
        device=0,       # This tells it to use your NVIDIA GPU
        batch=4,        # 3050 Ti has 4GB VRAM. '8' might crash it. '4' is safe.
        workers=2,      # Uses CPU cores to load data faster for the GPU
        
        name='yolo_medium_run', 
        
        # --- Augmentation ---
        degrees=10.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("Exporting...")
    model.export(format='onnx')

if __name__ == '__main__':
    main()