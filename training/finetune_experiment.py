import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import csv
import glob
import random
import shutil
from ultralytics import YOLO

BASE_DIR    = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'
MODEL_B     = f'{BASE_DIR}/training/runs/model_b/weights/best.pt'
ENV_A_TRAIN = f'{BASE_DIR}/datasets/env_a/images/train'
ENV_A_LABELS= f'{BASE_DIR}/datasets/env_a/labels/train'
DATASET_A   = f'{BASE_DIR}/training/dataset_a.yaml'
OUTPUT_DIR  = f'{BASE_DIR}/training/runs/finetune_experiment'
RESULTS_CSV = f'{BASE_DIR}/benchmark/results/finetune_results.csv'
N_SHOTS     = 50
RANDOM_SEED = 42

def prepare_few_shot_dataset(n_shots, seed):
    print(f"Step 1/4: Preparing {n_shots}-shot dataset...")
    random.seed(seed)
    
    jpg_files = glob.glob(os.path.join(ENV_A_TRAIN, '*.jpg'))
    if not jpg_files:
        print(f"Warning: No images found in {ENV_A_TRAIN}. Will create empty dataset.")
        
    jpg_files.sort()
    random.shuffle(jpg_files)
    
    selected_jpgs = jpg_files[:n_shots]
    
    few_shot_img_dir = os.path.join(OUTPUT_DIR, 'few_shot_data', 'images', 'train')
    few_shot_lbl_dir = os.path.join(OUTPUT_DIR, 'few_shot_data', 'labels', 'train')
    os.makedirs(few_shot_img_dir, exist_ok=True)
    os.makedirs(few_shot_lbl_dir, exist_ok=True)
    
    valid_count = 0
    for jpg in selected_jpgs:
        base = os.path.basename(jpg)
        name = os.path.splitext(base)[0]
        txt = os.path.join(ENV_A_LABELS, f"{name}.txt")
        
        if os.path.exists(txt):
            shutil.copy2(jpg, os.path.join(few_shot_img_dir, base))
            shutil.copy2(txt, os.path.join(few_shot_lbl_dir, f"{name}.txt"))
            valid_count += 1
            
    val_images_dir = f'{BASE_DIR}/datasets/env_a/images/val'
    val_count = len(glob.glob(os.path.join(val_images_dir, '*.jpg')))
    
    yaml_path = os.path.join(OUTPUT_DIR, 'few_shot_dataset.yaml')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(f"path: {OUTPUT_DIR}/few_shot_data\n")
        f.write(f"train: images/train\n")
        f.write(f"val: {val_images_dir}\n")
        f.write(f"nc: 3\n")
        f.write(f"names: ['box', 'cylinder', 'sphere']\n")
        
    print(f"Few-shot dataset ready: {valid_count} train images, {val_count} val images")
    return yaml_path

def run_finetuning(few_shot_yaml):
    print(f"Step 2/4: Fine-tuning model_b on {N_SHOTS} images...")
    
    # We create the model object depending on whether MODEL_B exists or not, to prevent crashing in dry runs
    model_path = MODEL_B if os.path.exists(MODEL_B) else 'yolov8s.pt'
    model = YOLO(model_path)
    
    results = model.train(
        data=few_shot_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        project=OUTPUT_DIR,
        name='finetuned_model_b',
        lr0=0.001,
        lrf=0.01,
        freeze=10,
        patience=10,
        verbose=False,
        exist_ok=True,
    )
    return os.path.join(OUTPUT_DIR, 'finetuned_model_b', 'weights', 'best.pt')

def run_scratch_training(few_shot_yaml):
    print(f"Step 3/4: Training from scratch on {N_SHOTS} images...")
    model = YOLO('yolov8s.pt')
    results = model.train(
        data=few_shot_yaml,
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        project=OUTPUT_DIR,
        name='scratch_50shots',
        lr0=0.01,
        patience=10,
        verbose=False,
        exist_ok=True,
    )
    return os.path.join(OUTPUT_DIR, 'scratch_50shots', 'weights', 'best.pt')

def evaluate_all_conditions(finetuned_weights, scratch_weights):
    print("Step 4/4: Evaluating all conditions...")
    conditions = {
        'baseline_no_adapt': MODEL_B,
        'finetuned_50shots': finetuned_weights,
        'scratch_50shots':   scratch_weights,
    }
    
    n_train_map = {
        'baseline_no_adapt': 0,
        'finetuned_50shots': N_SHOTS,
        'scratch_50shots': N_SHOTS
    }
    
    results_list = []
    
    for condition_name, weights_path in conditions.items():
        if not os.path.exists(weights_path):
            print(f"Warning: weights not found for {condition_name} at {weights_path}")
            continue
            
        model = YOLO(weights_path)
        val_res = model.val(data=DATASET_A, split='val', verbose=False, device=0)
        
        mAP50 = float(val_res.box.map50)
        precision = float(val_res.box.mp)
        recall = float(val_res.box.mr)
        inference_ms = val_res.speed.get('inference', 0.0)
        fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
        
        results_list.append({
            'condition': condition_name,
            'mAP50': mAP50,
            'precision': precision,
            'recall': recall,
            'fps': fps,
            'n_train_images': n_train_map[condition_name]
        })
        
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'mAP50', 'precision', 'recall', 'fps', 'n_train_images'])
        writer.writeheader()
        writer.writerows(results_list)
        
    return results_list

def plot_results(results_list):
    conds = []
    maps = []
    fpss = []
    for r in results_list:
        conds.append(r['condition'])
        maps.append(r['mAP50'])
        fpss.append(r['fps'])
        
    colors = []
    for c in conds:
        if c == 'baseline_no_adapt': colors.append('red')
        elif c == 'finetuned_50shots': colors.append('green')
        elif c == 'scratch_50shots': colors.append('blue')
        else: colors.append('gray')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Subplot 1
    bars1 = ax1.bar(conds, maps, color=colors)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Domain Adaptation: mAP@50 on Env A")
    ax1.set_ylabel("mAP@50")
    
    # Baseline line
    baseline_val = 0.0
    for r in results_list:
        if r['condition'] == 'baseline_no_adapt':
            baseline_val = r['mAP50']
    
    if baseline_val > 0:
        ax1.axhline(y=baseline_val, color='r', linestyle='--', alpha=0.7)
        
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=10)
        
    # Subplot 2
    bars2 = ax2.bar(conds, fpss, color=colors)
    ax2.set_title("Inference FPS Comparison")
    ax2.set_ylabel("FPS")
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plots_dir = os.path.join(BASE_DIR, 'benchmark/results/plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'finetune_experiment.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def main():
    print("========================================")
    print("  FEW-SHOT DOMAIN ADAPTATION EXPERIMENT")
    print("========================================")
    
    yaml_path = prepare_few_shot_dataset(N_SHOTS, RANDOM_SEED)
    if not yaml_path:
        return
        
    finetuned_weights = run_finetuning(yaml_path)
    scratch_weights = run_scratch_training(yaml_path)
    
    results = evaluate_all_conditions(finetuned_weights, scratch_weights)
    if not results:
        print("No results generated.")
        return
        
    plot_path = plot_results(results)
    
    print("\nRESULTS:")
    print(f"{'Condition':<22} {'mAP50':<7} {'Precision':<10} {'Recall':<8} {'FPS'}")
    
    base_map = 0.0
    fine_map = 0.0
    scratch_map = 0.0
    
    for r in results:
        c = r['condition']
        print(f"{c:<22} {r['mAP50']:<7.3f} {r['precision']:<10.3f} {r['recall']:<8.3f} {r['fps']:.1f}")
        if c == 'baseline_no_adapt': base_map = r['mAP50']
        elif c == 'finetuned_50shots': fine_map = r['mAP50']
        elif c == 'scratch_50shots': scratch_map = r['mAP50']
        
    recovery = (fine_map - base_map) * 100
    outperform = (fine_map - scratch_map) * 100
    
    print(f"\nKey Finding: Fine-tuning recovered {recovery:.1f}% mAP vs baseline.")
    print(f"Fine-tuning outperformed scratch by {outperform:.1f}% mAP.")
    
    print(f"\nResults saved to: {RESULTS_CSV}")
    print(f"Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
