from ultralytics import YOLO
import os
import wandb

def main():
    wandb.init(project="Lung-AI-Project", name="yolo_medium_advanced_v1")
    print("Loading YOLOv8-Medium model...")
    model = YOLO('yolov8m.pt') 

    yaml_path = os.path.join(os.getcwd(), 'datasets', 'lung_data', 'data.yaml')

    print(f"Training on data at: {yaml_path}")
    results = model.train(
        data=yaml_path,
        epochs=30,      
        imgsz=640,
        batch=4,        # Kept it safe for your GPU
        device=0,       
        
        # --- NEW "ADVANCED" PARAMETERS ---
        # These are based on our EDA
        optimizer='AdamW',  # A more stable optimizer than the default
        lr0=0.001,          # A lower learning rate to learn more carefully
        dropout=0.2,        # Helps prevent overfitting on smaller datasets
        #cls=...            # We will discuss this if your imbalance is severe

        name='yolo_medium_advanced_v1', # Give it a new name
        
        # --- Augmentation (keep this) ---
        degrees=10.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("Exporting...")
    model.export(format='onnx')

if __name__ == '__main__':
    main()