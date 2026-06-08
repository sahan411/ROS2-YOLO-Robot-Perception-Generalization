#!/usr/bin/env python3
import os
import glob
import argparse
import random
import cv2

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to dataset directory")
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print(f"FAIL: Directory does not exist: {dataset_dir}")
        return

    images = glob.glob(os.path.join(dataset_dir, '*.jpg'))
    total_images = len(images)
    
    if total_images == 0:
        print("FAIL: No images found in directory.")
        return

    matching_labels = 0
    missing_labels = []
    
    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    empty_labels = 0
    suspicious_boxes = 0
    
    # Track images that actually have labels for sampling
    images_with_boxes = []

    for img_path in images:
        base = os.path.splitext(img_path)[0]
        txt_path = base + '.txt'
        
        if os.path.exists(txt_path):
            matching_labels += 1
            has_boxes = False
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                if not lines or lines[0].strip() == '':
                    empty_labels += 1
                else:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c, cx, cy, w, h = parts
                            c = int(c)
                            w = float(w)
                            h = float(h)
                            
                            total_boxes += 1
                            if c in class_counts:
                                class_counts[c] += 1
                            else:
                                class_counts[c] = 1
                                
                            if w > 0.9 or h > 0.9:
                                suspicious_boxes += 1
                            has_boxes = True
            if has_boxes:
                images_with_boxes.append(img_path)
        else:
            missing_labels.append(img_path)
            
    print("=== Dataset Quality Report ===")
    print(f"Total images: {total_images}")
    print(f"Matching label files: {matching_labels}")
    print(f"Missing label files: {len(missing_labels)}")
    if missing_labels:
        print(f"  Example missing: {missing_labels[0]}")
    print(f"\nTotal bounding boxes: {total_boxes}")
    print(f"Per-class count:")
    for k, v in class_counts.items():
        print(f"  Class {k}: {v}")
    print(f"Empty label files (background): {empty_labels}")
    avg_boxes = total_boxes / total_images if total_images > 0 else 0
    print(f"Average boxes per image: {avg_boxes:.2f}")
    print(f"Suspiciously large boxes (w/h > 0.9): {suspicious_boxes}")
    
    # PASS/FAIL
    ratio = matching_labels / total_images
    passed = True
    reasons = []
    
    if ratio <= 0.8:
        passed = False
        reasons.append(f"Label ratio {ratio*100:.1f}% <= 80%")
    if total_boxes == 0:
        passed = False
        reasons.append("No bounding boxes detected total")
        
    print("\n=== Result ===")
    if passed:
        print("STATUS: PASS")
    else:
        print(f"STATUS: FAIL - {', '.join(reasons)}")
        
    # Sample 5 images and draw boxes
    sample_dir = os.path.join(dataset_dir, 'validation_samples')
    os.makedirs(sample_dir, exist_ok=True)
    
    class_names = ['box', 'cylinder', 'sphere']
    # Colors in BGR format: 0=red, 1=blue, 2=green
    colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0)}
    
    # Prefer images with boxes for sampling, otherwise pick any
    pool = images_with_boxes if len(images_with_boxes) >= 5 else images
    if len(pool) > 5:
        samples = random.sample(pool, 5)
    else:
        samples = pool
        
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_img, w_img, _ = img.shape
        txt_path = os.path.splitext(img_path)[0] + '.txt'
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() == '':
                        continue
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, cx, cy, w, h = parts
                        c = int(c)
                        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                        
                        # YOLO format is normalized center coordinates
                        box_w = w * w_img
                        box_h = h * h_img
                        box_cx = cx * w_img
                        box_cy = cy * h_img
                        
                        x1 = int(box_cx - box_w / 2)
                        y1 = int(box_cy - box_h / 2)
                        x2 = int(box_cx + box_w / 2)
                        y2 = int(box_cy + box_h / 2)
                        
                        color = colors.get(c, (255, 255, 255))
                        name = class_names[c] if c < len(class_names) else str(c)
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
        out_path = os.path.join(sample_dir, f'check_{i+1:05d}.jpg')
        cv2.imwrite(out_path, img)
        
    print(f"\nSaved {len(samples)} validation samples with drawn boxes to {sample_dir}")

if __name__ == '__main__':
    main()
