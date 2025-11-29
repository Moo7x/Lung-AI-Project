from ultralytics import YOLO
import os

def main():
    # 1. Load the Medium Model
    model = YOLO('yolov8m.pt') 

    # 2. Define Path
    yaml_path = os.path.join(os.getcwd(), 'datasets', 'lung_data', 'data.yaml')

    # 3. Start Advanced Training
    print("Starting Advanced Training (Optimized for Multitasking)...")
    
    results = model.train(
        data=yaml_path,
        
        # --- TIME SETTINGS ---
        epochs=40,           # A bit more than baseline to allow learning
        patience=5,          # STRICT: If no improvement for 5 epochs, STOP.
        
        # --- HARDWARE ---
        imgsz=640,
        batch=4,             # Safe for 4GB VRAM
        device=0,
        workers=1,           # REDUCED to 1 so your laptop doesn't lag while you do homework
        
        # --- ENHANCEMENTS ---
        optimizer='AdamW',   # The "Pro" optimizer
        lr0=0.001,           
        dropout=0.2,         
        cos_lr=True,         
        
        # --- AUGMENTATION ---
        degrees=10.0,        
        fliplr=0.5,          
        mosaic=1.0,          
        
        # --- OUTPUT ---
        name='yolo_medium_advanced_v1',
        save=True
    )

    # 4. Export
    model.export(format='onnx')

if __name__ == '__main__':
    main()