from ultralytics import YOLO
import os

def main():
    # 1. Point to HER model
    # (Make sure you renamed it to teammate_model.pt and put it in src/models)
    model_path = os.path.join("src", "models", "teammate_model.pt")
    
    print(f"Loading Teammate's Model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception:
        print("Error: Could not find teammate_model.pt in src/models/")
        return

    # 2. Point to YOUR data
    yaml_path = os.path.join("datasets", "lung_data", "data.yaml")

    # 3. Run Validation on the TEST set
    print("Running validation on Teammate's model...")
    metrics = model.val(data=yaml_path, split='test')

    print("\n--- TEAMMATE RESULTS ---")
    print(f"mAP50 (Accuracy): {metrics.box.map50}")
    print("------------------------")

if __name__ == "__main__":
    main()