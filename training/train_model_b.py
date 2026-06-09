#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO

def main():
    print("--- GPU Information ---")
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available! Using CPU.")
    print("-----------------------\n")

    base_dir = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'
    data_yaml = os.path.join(base_dir, 'training/dataset_b.yaml')
    project_dir = os.path.join(base_dir, 'training/runs')

    print("Loading YOLOv8s model...")
    model = YOLO('yolov8s.pt')

    print(f"Starting training for model_b using {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        project=project_dir,
        name='model_b',
        save=True,
        plots=True,
        patience=20,
        workers=8
    )

    print("\nTraining complete!")
    best_model_path = os.path.join(project_dir, 'model_b', 'weights', 'best.pt')
    print(f"Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()
