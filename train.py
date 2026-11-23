import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano model for transfer learning
    model = YOLO("yolov8n.pt")

    # Train on the custom helmet dataset
    results = model.train(
        data="dataset/data.yaml",
        epochs=10,          # Good baseline for CPU; increase to 50 on GPU
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,          # Prevents Windows multiprocessing RuntimeError
        project="runs/detect",
        name="helmet_yolov8n",
        exist_ok=True
    )

    # Copy best weights into models/helmet_best.pt for easy access
    best_weights = Path("runs/detect/helmet_yolov8n/weights/best.pt")
    target_path = Path("models/helmet_best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, target_path)
        print(f"\nTraining complete! Best model saved to: {target_path.resolve()}")

if __name__ == "__main__":
    main()
