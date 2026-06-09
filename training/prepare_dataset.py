#!/usr/bin/env python3
import os
import glob
import shutil
import random
import argparse

BASE_DIR = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'

def process_env(env_name, dataset_root):
    env_dir = os.path.join(dataset_root, f"env_{env_name}")
    if not os.path.isdir(env_dir):
        print(f"Directory not found: {env_dir}")
        return []
    
    jpg_files = glob.glob(os.path.join(env_dir, '*.jpg'))
    valid_pairs = []
    
    for jpg in jpg_files:
        txt = os.path.splitext(jpg)[0] + '.txt'
        if os.path.exists(txt):
            valid_pairs.append((jpg, txt))
        else:
            print(f"Warning: No matching label for {os.path.basename(jpg)}")
            
    # Sort for deterministic processing before shuffling
    valid_pairs.sort(key=lambda x: x[0])
    return valid_pairs

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset folder structure")
    parser.add_argument('--env', required=True, help="Environment name: a, b, c, or mixed")
    parser.add_argument('--split', type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    
    random.seed(args.seed)
    dataset_root = os.path.join(BASE_DIR, 'datasets')
    
    if args.env == 'mixed':
        pairs_a = process_env('a', dataset_root)
        pairs_b = process_env('b', dataset_root)
        pairs_c = process_env('c', dataset_root)
        
        min_count = min(len(pairs_a), len(pairs_b), len(pairs_c))
        if min_count == 0:
            print("Error: One or more environments have 0 valid pairs. Cannot create balanced mixed dataset.")
            return
            
        random.shuffle(pairs_a)
        random.shuffle(pairs_b)
        random.shuffle(pairs_c)
        
        combined_pairs = pairs_a[:min_count] + pairs_b[:min_count] + pairs_c[:min_count]
        out_env_dir = os.path.join(dataset_root, 'mixed')
    else:
        combined_pairs = process_env(args.env, dataset_root)
        out_env_dir = os.path.join(dataset_root, f"env_{args.env}")
        
    total_pairs = len(combined_pairs)
    if total_pairs == 0:
        print(f"No valid data pairs found for env: {args.env}")
        return
        
    random.shuffle(combined_pairs)
    
    train_count = int(total_pairs * args.split)
    val_count = total_pairs - train_count
    
    train_pairs = combined_pairs[:train_count]
    val_pairs = combined_pairs[train_count:]
    
    # Create directories
    img_train_dir = os.path.join(out_env_dir, 'images', 'train')
    img_val_dir = os.path.join(out_env_dir, 'images', 'val')
    lbl_train_dir = os.path.join(out_env_dir, 'labels', 'train')
    lbl_val_dir = os.path.join(out_env_dir, 'labels', 'val')
    
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)
    
    # Copy files
    print("Copying files to train/val directories...")
    for jpg, txt in train_pairs:
        # Avoid copying to itself if somehow output is same as input
        if os.path.dirname(jpg) != img_train_dir:
            shutil.copy2(jpg, os.path.join(img_train_dir, os.path.basename(jpg)))
            shutil.copy2(txt, os.path.join(lbl_train_dir, os.path.basename(txt)))
        
    for jpg, txt in val_pairs:
        if os.path.dirname(jpg) != img_val_dir:
            shutil.copy2(jpg, os.path.join(img_val_dir, os.path.basename(jpg)))
            shutil.copy2(txt, os.path.join(lbl_val_dir, os.path.basename(txt)))
        
    # Print summary
    env_display = args.env.upper() if args.env != 'mixed' else 'Mixed'
    print(f"Env {env_display}: {total_pairs} total -> {train_count} train, {val_count} val")
    print(f"Labels: {train_count} train labels, {val_count} val labels")
    print("Ready for training.")

if __name__ == '__main__':
    main()
