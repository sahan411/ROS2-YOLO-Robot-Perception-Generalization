import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import csv
import numpy as np
from ultralytics import YOLO

BASE_DIR = '/home/sdvn_hinder_metric/Sahan/ROS2-YOLO-Robot-Perception-Generalization'

MODELS = {
    'model_a':     f'{BASE_DIR}/training/runs/model_a/weights/best.pt',
    'model_b':     f'{BASE_DIR}/training/runs/model_b/weights/best.pt',
    'model_mixed': f'{BASE_DIR}/training/runs/model_mixed/weights/best.pt',
}

DATASETS = {
    'env_a': f'{BASE_DIR}/training/dataset_a.yaml',
    'env_b': f'{BASE_DIR}/training/dataset_b.yaml',
    'env_c': f'{BASE_DIR}/training/dataset_c.yaml',
}

RESULTS_DIR = f'{BASE_DIR}/benchmark/results'
PLOTS_DIR   = f'{BASE_DIR}/benchmark/results/plots'

def evaluate_model(model_name, model_path, dataset_name, dataset_yaml):
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found: {model_path}")
        return None
        
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=dataset_yaml, split='val', verbose=False, device=0)
    
    # Extract metrics
    mAP50 = float(results.box.map50)
    precision = float(results.box.mp)
    recall = float(results.box.mr)
    
    # Calculate fps handling potential ultralytics api variations
    inference_ms = results.speed.get('inference', 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    
    is_baseline = False
    if model_name == 'model_a' and dataset_name == 'env_a':
        is_baseline = True
    elif model_name == 'model_b' and dataset_name == 'env_b':
        is_baseline = True
        
    print(f"✓ {model_name} vs {dataset_name} | mAP50: {mAP50:.3f} | FPS: {fps:.1f}")
    
    return {
        'model': model_name,
        'test_env': dataset_name,
        'mAP50': mAP50,
        'precision': precision,
        'recall': recall,
        'fps': fps,
        'is_baseline': is_baseline
    }

def save_results(results_list):
    csv_path = os.path.join(RESULTS_DIR, 'matrix_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'test_env', 'mAP50', 'precision', 'recall', 'fps', 'is_baseline'])
        writer.writeheader()
        writer.writerows(results_list)
        
    models = list(MODELS.keys())
    envs = list(DATASETS.keys())
    
    table_dict = {m: {e: "" for e in envs} for m in models}
    for r in results_list:
        val = f"{r['mAP50']:.3f}"
        if r['is_baseline']:
            val += "*"
        table_dict[r['model']][r['test_env']] = val
        
    summary_path = os.path.join(RESULTS_DIR, 'matrix_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("╔══════════════════════════════════════════════════════╗\n")
        f.write("║           BENCHMARK MATRIX RESULTS                   ║\n")
        f.write("╠══════════════════════════════════════════════════════╣\n")
        f.write("║              │  Test EnvA │  Test EnvB │  Test EnvC  ║\n")
        f.write("╠══════════════════════════════════════════════════════╣\n")
        
        for m in models:
            m_disp = m.replace('_', ' ').title()
            if m_disp == 'Model Mixed':
                m_disp = 'Model Mix'
            val_a = table_dict[m].get('env_a', '   -   ')
            val_b = table_dict[m].get('env_b', '   -   ')
            val_c = table_dict[m].get('env_c', '   -   ')
            f.write(f"║ {m_disp[:12]:<12} │   {val_a:<8} │   {val_b:<8} │   {val_c:<8}  ║\n")
            
        f.write("╠══════════════════════════════════════════════════════╣\n")
        f.write("║ * = baseline (trained and tested on same env)        ║\n")
        f.write("╚══════════════════════════════════════════════════════╝\n")
        
    with open(summary_path, 'r') as f:
        print("\n" + f.read())

def plot_heatmaps(results_dict):
    models = list(MODELS.keys())
    envs = list(DATASETS.keys())
    
    map_matrix = np.zeros((len(models), len(envs)))
    shift_matrix = np.zeros((len(models), len(envs)))
    
    baseline_map = {}
    for m in models:
        for e in envs:
            if (m, e) in results_dict and results_dict[(m, e)]['is_baseline']:
                baseline_map[m] = results_dict[(m, e)]['mAP50']
    
    for i, m in enumerate(models):
        for j, e in enumerate(envs):
            if (m, e) in results_dict:
                map_val = results_dict[(m, e)]['mAP50']
                map_matrix[i, j] = map_val
                b_map = baseline_map.get(m, map_val)
                shift_matrix[i, j] = b_map - map_val
            else:
                map_matrix[i, j] = 0.0
                shift_matrix[i, j] = 0.0

    # Heatmap 1 - mAP50 matrix
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    cax = ax.matshow(map_matrix, cmap='RdYlGn')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(envs)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(envs)
    ax.set_yticklabels(models)
    
    for i in range(len(models)):
        for j in range(len(envs)):
            val = map_matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black" if 0.3 < val < 0.7 else "white")
            
    ax.set_title("mAP@50 Cross-Evaluation Matrix", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_mAP50.png'))
    plt.close()
    
    # Heatmap 2 - Domain shift
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    cax = ax.matshow(shift_matrix, cmap='RdYlGn_r')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(envs)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(envs)
    ax.set_yticklabels(models)
    
    for i in range(len(models)):
        for j in range(len(envs)):
            val = shift_matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black" if -0.2 < val < 0.2 else "white")
            
    ax.set_title("Domain Shift Drop from Baseline", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'heatmap_domain_shift.png'))
    plt.close()

def plot_bar_charts(results_dict):
    models = list(MODELS.keys())
    envs = list(DATASETS.keys())
    
    # Chart 1 - Grouped mAP50
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(envs))
    width = 0.25
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, m in enumerate(models):
        y_vals = []
        for e in envs:
            if (m, e) in results_dict:
                y_vals.append(results_dict[(m, e)]['mAP50'])
            else:
                y_vals.append(0.0)
        ax.bar(x + (i-1)*width, y_vals, width, label=m, color=colors[i])
        
    ax.set_ylabel('mAP@50')
    ax.set_title('Model Performance Across Test Environments')
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bar_mAP50.png'))
    plt.close()
    
    # Chart 2 - FPS comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    labels = []
    fps_vals = []
    
    for m in models:
        for e in envs:
            if (m, e) in results_dict:
                labels.append(f"{m}\n{e}")
                fps_vals.append(results_dict[(m, e)]['fps'])
                
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, fps_vals, color='tab:purple')
    ax.set_ylabel('FPS')
    ax.set_title('Inference FPS per Combination')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'bar_fps.png'))
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    results_list = []
    results_dict = {}
    
    for model_name, model_path in MODELS.items():
        for dataset_name, dataset_yaml in DATASETS.items():
            res = evaluate_model(model_name, model_path, dataset_name, dataset_yaml)
            if res:
                results_list.append(res)
                results_dict[(model_name, dataset_name)] = res
                
    if results_list:
        save_results(results_list)
        plot_heatmaps(results_dict)
        plot_bar_charts(results_dict)

if __name__ == '__main__':
    main()
