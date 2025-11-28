from ultralytics import YOLO
import os

def main():
    # 1. Point to the saved file from your successful training
    # Note: The log you sent says it saved to 'yolo_medium_run2'
    # We want the 'best.pt' file inside that folder.
    
    weights_path = os.path.join(os.getcwd(), 'runs', 'detect', 'yolo_medium_run2', 'weights', 'best.pt')
    
    print(f"Loading weights from: {weights_path}")
    
    # 2. Load the model
    model = YOLO(weights_path)

    # 3. Export to ONNX
    print("Exporting to ONNX...")
    model.export(format='onnx')
    print("Success! Export complete.")

if __name__ == '__main__':
    main()